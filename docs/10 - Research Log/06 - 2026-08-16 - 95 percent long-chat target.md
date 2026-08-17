# Locked 95% long-chat accuracy target

**Date**: 2026-08-16
**Status**: ACTIVE TARGET — measurement gate built; target not yet achieved
**Primary benchmark**: cleaned LongMemEval oracle, then LongMemEval-S
**Secondary benchmark**: LoCoMo

> **Dataset correction, 2026-08-16:** the original 200-question development
> trace below was built from `longmemeval_oracle.json`, despite being discussed
> as the long-haystack corpus. Those figures are oracle diagnostics, not
> LongMemEval-S results. The current official cleaned LongMemEval-S file is
> 277,383,467 bytes with SHA-256
> `d6f21ea9d60a0d56f34a05b609c79c88a451d2ae03597821ea3d5a9678c3a442`.
> All current long-haystack work uses split manifest v2 bound to that hash.

## Target

The primary test is operational: ingest a completed set of turns, ask a later
question, send only the bounded memory-produced context to the responder, and
grade the resulting answer. The whole transcript is counted for compression
but is never placed in the treatment prompt.

memory_condense must reach **at least 95% answer-stage judge accuracy** on this
locked long-chat evaluation while respecting all of these constraints:

- at least 100 judge-graded questions (200 on final LongMemEval confirmation);
- at most 8,000 cl100k proxy tokens in each responder prompt;
- identical responder and judge models across baseline/treatment comparisons;
- exact source provenance retained;
- zero persisted transformer token K/V or residual sequences; and
- report accuracy, mean/p95 prompt tokens, write cost, read latency, and storage
  rather than optimizing a single headline number.

Accuracy is the hard gate. Token saving, latency, heat balance, and pruning are
optimized only among arms that meet it.

Every benchmark report therefore records the mean completed-transcript token
count, retrieved-context token count, fraction sent, percentage saved, maximum
prompt size, and whether every question stayed within the configured prompt
budget. A 95% judge score cannot pass if any prompt exceeded that boundary.
Evidence-source coverage and literal containment remain failure diagnostics;
neither is the operational outcome.

## Why answer containment is not the target

The existing free diagnostic asks whether the gold answer string occurs in the
retrieved text. This is valuable for failure analysis but it is not long-chat
accuracy: temporal, multi-session, update, and open-domain questions often
require inference or paraphrase. LoCoMo's measured verbatim ceiling is far
below 95%. The target therefore uses semantic judge correctness; F1, exact
match, and containment remain secondary diagnostics.

## Locked population

The official cleaned `longmemeval_oracle.json` was downloaded from the
[LongMemEval repository's documented Hugging Face release](https://github.com/xiaowu0162/LongMemEval/blob/main/README.md).

- File size: 15,388,478 bytes
- SHA-256:
  `821a2034d219ab45846873dd14c14f12cfe7776e73527a483f9dac095d38620c`
- Population: 500 samples / 500 questions
- Categories: 78 knowledge update, 133 multi-session, 56 assistant fact,
  30 preference, 70 user fact, 133 temporal reasoning

The manifest
`data/longmemeval-95-target-split-v1.json` in this research-log directory
uses deterministic largest-remainder apportionment by question category, then
orders samples inside each category by
`sha256(salt + NUL + category + NUL + sample_id)`. A second independent
`sha256(salt + NUL + "order" + NUL + sample_id)` orders each completed
partition so capped smoke runs are not category blocks:

| Partition | Questions | Use |
| --- | ---: | --- |
| Development | 200 | failure analysis and tuning |
| Validation | 100 | policy selection; limited looks |
| Confirmation | 200 | one final locked target decision |

The CLI verifies the dataset hash before selecting a partition. Dataset order
cannot move a sample between partitions.

## Implemented measurement gate

The benchmark report now records per-question context and full prompt-content
tokens, aggregate mean/p95 prompt tokens, the configured accuracy target,
minimum question count, and an explicit target status:

- `ungraded` — no semantic judge;
- `insufficient_questions` — judged but below the minimum population;
- `failed` — sufficiently sized but below 95%; or
- `passed` — sufficiently sized and at or above 95%.

Before an answer call, ranked excerpts are fitted under the 8,000-token cap.
The final excerpt is token-boundary truncated if needed; the fully assembled
prompt is re-counted, so the ceiling is enforced rather than estimated.

Example development command:

```powershell
pixi run --frozen -e dev python -m memory_condense.eval `
  --benchmark-file data/longmemeval_oracle.json `
  --benchmark-format longmemeval `
  --benchmark-split-manifest "docs/10 - Research Log/data/longmemeval-95-target-split-v1.json" `
  --benchmark-split development `
  --mode span --use-judge `
  --accuracy-target 0.95 --min-target-questions 100 `
  --max-prompt-tokens 8000
```

This command makes paid answer and judge calls. Run the same partition through
`--answer-recall` first to reject obviously weak retrieval arms for free.

## Competitive threshold

Mem0 currently reports 91.6% LoCoMo and 93.4% LongMemEval accuracy at 6,956
and 6,787 mean tokens, respectively. Those are managed-Platform vendor results
with a stated ±1 point judge interval and proprietary optimizations; OSS parity
is not promised. See Mem0's
[current official evaluation documentation](https://docs.mem0.ai/core-concepts/memory-evaluation).

Consequently, 95% at an 8k ceiling is intentionally ambitious but meaningful:
it is above the current Mem0 LongMemEval point result, while leaving enough
budget to test whether QK/heat allocation can beat shallow top-k retrieval.

## Immediate experiment order

1. Free development reachability: hybrid, pooled span, ranked QK, dual QK/heat,
   and degree-two dual QK/heat.
2. Paid development answers only for Pareto-relevant retrieval arms.
3. Failure clusters by category and evidence-source miss; change write-time
   association coverage before increasing graph depth.
4. One validation selection.
5. One 200-question confirmation run with frozen code/config hashes.

The current generic public-benchmark harness does not yet compile or load a
per-sample QK/heat artifact. That integration, plus reusable compiled sample
stores, is the next engineering step. Until it exists, the 95% target is
measurable for dense/hybrid/memory/span but not yet for the new association
policy.

## First development preflight

Before any answer or judge calls, the first 20 samples in the deterministically
mixed development order were measured under the 8k ceiling:

| Arm | Gold-string reachability | Mean context tokens | Reachability / 1k |
| --- | ---: | ---: | ---: |
| Pooled spans, 110/220 tokens × 2 | 40.0% | 294 | 136.17 |
| Hybrid `k=10` | **45.0%** | 1,285 | 35.01 |
| Hybrid `k=50` | **45.0%** | 4,763 | 9.45 |

The sample is too small for an accuracy conclusion and containment is not the
headline metric. It is already enough to reject indiscriminate widening:
`k=50` spent 3.7× the hybrid `k=10` context without making another gold string
reachable. The category diagnostic was strongest on direct user facts (100%
for hybrid) and weakest on multi-session (0%) and temporal reasoning (14.3%).
That failure shape motivates learned turn-transition/source links rather than a
larger flat top-k.

The preflight also found and fixed a major harness performance defect. A fresh
`EmbeddingService` was constructed for each one-question LongMemEval sample,
reloading bge-m3 repeatedly. A 20-question run did not finish within ten
minutes. Reusing one stateless embedder while retaining isolated SQLite/HNSW
stores completed the same-sized run in about 60 seconds. Model weights are now
shared; benchmark memory state is not.

## Source-identity correction and matched preflight

The first preflight silently discarded LongMemEval `haystack_dates`,
`question_date`, and durable session identity. That made temporal probes less
answerable than the source data and made source-balanced retrieval impossible
to evaluate. Schema v6 now stores `turns.source_id`; loaders add source-tagged
session timestamp turns and the dated question is used for retrieval and QA.

The same locked first 20 development questions were rerun under the 8k cap.
Evidence-source metrics use the benchmark's `answer_session_ids` and report
mean fractional coverage plus any/all-source rates:

| Arm | Gold-string reachability | Evidence coverage | All evidence | Mean context tokens |
| --- | ---: | ---: | ---: | ---: |
| Whole-source pooling, 4 sources | 45.0% | 95.1% | 85.0% | 4,886 |
| Hybrid chunks, `k=10` | 45.0% | **100.0%** | **100.0%** | **1,090** |

These are oracle-haystack diagnostics, not answer-stage accuracy. The result
rejects whole-source replacement: it spent 4.5× the context and covered fewer
complete evidence sets. The useful role for turn-transition learning is now
narrower and testable—gate/rerank attention inside compact hybrid candidates,
not inflate the prompt with full sessions.

Artifacts:

- `C:\Users\Keytone\Downloads\memory-condense-rig\longmemeval-95-source-k4-20.csv`
- `C:\Users\Keytone\Downloads\memory-condense-rig\longmemeval-95-hybrid-k10-20.csv`

## Delayed turn-transition learner

`transition_policy.py` now enforces the causal event order directly:

1. `propose` freezes a bounded decision at turn `t` and performs no update.
2. `observe` accepts turn `t+1`, computes its CAV delta, rewards target edges,
   and optionally weights each head by projected-OV/delta cosine alignment.
3. user→assistant and assistant→user statistics remain disjoint.
4. snapshots contain only decayed scalar reward/mass/count statistics and
   source/destination IDs; pending CAV vectors, text, token K/V, and residual
   sequences are excluded.

This is an implemented mechanism, not evidence of a recall gain. Its next gate
is a chronological replay whose teacher target at `t+1` is revealed only after
the `t` ranking. Until it beats ungated QK at the same candidate and token
budgets, transition utility cannot control QA retrieval or deletion.

### Causal replay result

The gate was run against two independently compiled real stores. Exact
next-chunk prediction was mostly undefined because the current narrow graph
contained the target only 3.6% of the time on the first store. A global CAV
transition weight and a CAV-velocity control improved development delta cosine,
but both failed transfer: on the second store the selected global transition
weight changed delta cosine from 0.248 to 0.217, while the selected velocity
control changed it from 0.255 to 0.201. Exact R@1 did not improve. The current
two-dimensional CAV bank (`context_dependency`, `binding_constraint`) is too
coarse for routing, so learned transition weights remain excluded from reads
and pruning.

## Full development retrieval audit

The locked 200-question development split was measured with hybrid `k=10` and
the 8k hard gate:

- answer-string reachability: 43.5%;
- full-haystack literal ceiling: 47.5%;
- mean context: 1,043 tokens;
- mean/all/any evidence-source coverage: 100%; and
- runtime: 280.7 seconds after batched ingestion.

Thus the correct evidence session is already activated for every development
question; only eight questions have a normalized literal answer somewhere in
the haystack but not in the hybrid context. Whole-source hydration was rejected
on the first 20 questions because it added no literal recall while raising mean
context to 4,885 tokens. The bottleneck is evidence selection *inside* an
activated source, not source discovery.

`ingest_many` now appends/chunks a non-extracting sample and performs one batched
embedding/index update. The same 20-question hybrid output remained identical
while runtime fell from 64.4 to 50.8 seconds. Extracting mode deliberately keeps
the causal sequential path.

## Bounded source-local transition experiment

`hybrid_neighbor` preserves the hybrid top-10 anchors, then walks source-local
chunk shells in distance order. It has independent hard caps for radius and
extra neighbor slots. A fixed-budget variant lets transition candidates replace
the weakest anchors. No transformer is loaded and no token state is retained.

| Development arm | Literal reachability | Mean context tokens | Reachability / 1k | Row changes vs hybrid |
| --- | ---: | ---: | ---: | --- |
| Hybrid `k=10` | 43.5% | 1,043 | 41.71 | baseline |
| Radius 1, unlimited shell | 44.5% | 2,219 | 20.05 | +2 / -0 |
| Radius 1, 5 extra slots | 44.5% | 1,737 | 25.62 | +2 / -0 |
| Radius 1, replace 5 anchors | 40.0% | 1,158 | 34.54 | +2 / -9 |

The two positive transitions were a temporal-reasoning answer at neighbor slot
2 and a knowledge-update answer at slot 5. The remaining recoverable misses
needed radii 2, 3, 4, or 6; one was still unreachable at radius 6. Adjacency is
a real signal, but unconditional expansion is not Pareto-efficient and global
replacement is destructive. The next policy must choose among `stay`, previous,
next, bridge, and source-switch actions per query, with delayed token-normalized
reward. Validation remains untouched.

Artifacts:

- `C:\Users\Keytone\Downloads\memory-condense-rig\longmemeval-95-hybrid-k10-development-200.csv`
- `C:\Users\Keytone\Downloads\memory-condense-rig\longmemeval-95-hybrid-neighbor-r1-development-200.csv`
- `C:\Users\Keytone\Downloads\memory-condense-rig\longmemeval-95-hybrid-neighbor-r1-s5-development-200.csv`
- `C:\Users\Keytone\Downloads\memory-condense-rig\longmemeval-95-hybrid-neighbor-r1-replace5-development-200.csv`

### Reusable stores and frozen transition trace

Repeated development arms previously rebuilt every isolated sample store and
re-embedded its full haystack. `--compiled-store-cache` now creates one
content-addressed SQLite/HNSW store per sample. Its key covers the complete
sample payload, chunker, embedding identity, and schema version. A manifest
records exact SQLite and ANN hashes; both are verified on every cache hit.
Reads disable index persistence so evaluation cannot mutate the artifact.
Publication is directory-atomic where supported and uses a Windows-safe
manifest-last fallback otherwise, so a partial store is never a valid hit.

On the first 20 development questions, cold construction took 49.4 seconds and
a verified cache hit took 22.5 seconds. The two CSVs are byte-identical.
Query embedding now dominates the cached path.

The corrected development candidate trace batch-encodes all 200 questions once, then
records direct anchors and radius-six source-local candidates with direction,
distance, anchor rank, exact turn/source provenance, and rendered text. It
contains no transformer state. Internal trace SHA-256:
`90d68b1c21a2394d4d171710ebd50909d65460a864657175118944ec6bb96702`.

A 523-arm offline sweep completed in 10.5 seconds after exact rendered-token
counts replaced repeated prompt fitting for compositions provably below 8k.
The wider radius-six boundary found a monotonic accuracy-first candidate:

| Arm | Literal recall | Mean context tokens | Evidence coverage | Changes vs hybrid |
| --- | ---: | ---: | ---: | --- |
| Hybrid stay, `k=10` | 43.5% | 1,037.2 | 100% | baseline |
| Add radius 6 / 20 slots | 46.5% | 3,583.6 | 100% | +6 / -0 |
| **Add radius 6 / 23 slots** | **47.0%** | **3,817.5** | **100%** | **+7 / -0** |

The selected arm reaches every literal answer reachable by this bounded action
family and remains 44.3% below Mem0's reported 6,787 mean LongMemEval tokens.
One normalized haystack literal remains unreachable. This is still a retrieval
diagnostic, not evidence of 95% semantic answer accuracy. The frozen
development selection is
`data/longmemeval-development-selected-transition-v2.json`; validation remains
untouched.

The v1 selection was superseded after finding that sentence-level token counts
were summed before chunks were rejoined, undercounting rendered BPE tokens.
Chunk counts and maximums are now exact, Unicode hard splits preserve text, and
compiled-store cache revision 2 forces a rebuild. The corrected sweep retained
the same seven monotonic recoveries; only the precise slot count and token cost
changed.

The remote benchmark path is now spend-auditable before calibration. It refuses
all provider calls by default, requires `--max-provider-calls` to cover the
logical answer/judge count, defaults retries to zero, and persists actual usage
for both roles. A matched 20-question baseline/treatment calibration requires
exactly 80 logical calls with no retries. Offline prompt volume is 105,349
answer-input tokens across the two arms; conservative maxima are 10,240 answer
output, 14,878 judge input, and 40,960 judge output tokens. No such calls have
been authorized or made.

Benchmark reports now bind a paid result to exact evidence: dataset hash,
locked-split-manifest hash, complete `src/memory_condense/**/*.py` tree hash,
`pixi.lock` hash, and frozen-policy-manifest hash. Each JSON report receives a
SHA-256 sidecar. A green metric without these identities is not admissible as
the locked 95% result.

## Local responder boundary

A complete deterministic Qwen3 responder path was added for the local BF16
checkpoint, with explicit GPU/CPU placement ceilings and ephemeral generation
K/V. On the available 8 GB GPU plus CPU offload, even a one-question run did
not finish model placement within ten minutes. The process was terminated and
no score was produced. The full 36-layer model is therefore not a practical
local answerer on this host; the seven-layer bounded prefix remains an offline
linker. No paid answer or judge calls have been made.

## Official LongMemEval-S retrieval correction

The first 40 questions of the locked 200-question development split were used
as an implementation preflight. Validation was not opened. This is a free
literal/evidence diagnostic, not the 95% semantic-answer result.

The original hybrid context reached 37.5% of questions at 653 mean tokens. A
wide local-transition sweep found a 47.5% arm at 3,861 tokens, but a failure
audit separated five candidate-boundary misses from sixteen questions whose
gold answer is not a literal substring of the haystack. Four genuine literal
spans were at hybrid ranks 30, 57, 68, and 91; the fifth apparent span was an
unrelated decoy and its answer must be computed from dated evidence.

`search_hybrid_graph` now composes three independently bounded operations:

1. keep the normal top-10 hybrid anchors unchanged;
2. admit at most 24 next-direction source neighbors within radius five; and
3. let the top 20 candidates activate source IDs, then admit at most 48
   source-conditioned candidates from a 200-item pool.

It persists no transformer/token state and hydrates text only for final
candidates. Under the unchanged 8,000-token prompt ceiling:

| Development preflight arm (n=40) | Literal recall | Mean tokens | Mean evidence-source coverage | All evidence sources |
| --- | ---: | ---: | ---: | ---: |
| Hybrid `k=10` | 37.5% | 653 | 92.5% | 85.0% |
| Best local transition | 47.5% | 3,861 | 92.5% | 85.0% |
| Graph union, activation 10 / source 24 | 52.5% | 6,205 | 92.5% | 85.0% |
| Graph union, activation 20 / source 32 | 55.0% | 6,582 | 99.0% | 97.5% |
| **Graph union, activation 20 / source 48** | **57.5%** | **7,302** | **99.0%** | **97.5%** |

The accuracy-first arm reaches 23 of the 24 literal-answer questions and gains
four over the prior transition arm with zero losses. It retrieves at least one
gold evidence source for all 40 questions and every evidence source for 39.
The remaining temporal chain retrieves three of five evidence sessions and
does include the literal answer. Therefore retrieval is no longer the dominant
unknown on this preflight; semantic responder/judge accuracy remains unmeasured
and the 95% goal is not claimed.

Primary artifacts are under
`C:\Users\Keytone\Downloads\memory-condense-rig`, especially
`longmemeval-official-development-40-hybrid-graph-a20-s48-v2.csv` (SHA-256
`2671a6a92c64abbde95b719ec336cd7a519b31290f7d4553acb323a5384a4b63`).
The token-first `s32` CSV is
`ee85063e2f412cda750db3de91482d77fb8afd4bb2d5ee2bccc6b93f5cc06404`;
the failure-audit CSV is
`2bf16f3ee6e072fe3d79060b3123df8c7ecb7aca2e6c6e117133bd5e7375d18c`.

## Live causal-consolidation LongMemEval-S preflight

Schema-v9 causal replay is now integrated into isolated LongMemEval stores.
On the same locked first-40 development preflight, the frozen causal-graph arm
matches the previous wide arm's 57.5% literal recall row-for-row while reducing
mean retrieved context from 7,302 to 6,638 tokens. Mean evidence-source
coverage is 99.5%; all expected sources are present for 39/40 questions.

This moves the bottleneck but does not pass the target. Source coverage means
the correct sessions were activated; it does not guarantee that the right
within-session chunk survived packing or that a responder correctly performs
temporal, counting, update, and multi-session reasoning. Only 24/40 answers
have a normalized literal span anywhere in the haystack, and 23 are retrieved.
Answer-stage accuracy remains unmeasured because no provider calls were
authorized or made.

Implementation, performance corrections, cache identity, and the frozen
development policy are recorded in
`10 - 2026-08-16 - LongMemEval live consolidation preflight.md`.

An explicit partition-local source scan was subsequently implemented and
tested on the same 40 development questions. After correcting an invalid
per-source normalization that caused one temporal regression, the globally
calibrated local arm matched all 23 baseline literal hits but used 29 more mean
tokens and additional scan work. It is therefore retained as an ablation, not
selected. The only remaining haystack-literal discrepancy is a coincidental
"two weeks" outside the gold evidence; the gold sessions themselves require
date arithmetic and are both retrieved. This strengthens the case that the
next gate is answer-stage reasoning rather than broader search.

The follow-up gold-source audit makes that conclusion measurable. On the same
40 questions, only 50% of capped labelled sessions contain a normalized literal
answer; the other 50% require inference, aggregation, or semantic paraphrase.
The selected context retains 52.5%
literal containment specifically inside retrieved gold-source excerpts.
Therefore 99.5% source coverage is a routing metric, not 99.5% answerable
context and certainly not 99.5% answer accuracy. The new sufficiency harness
can judge the gold-source oracle and retrieved context independently once a
judge is authorized.

Every question with a literal answer in its capped gold sources is already a
retrieved-gold-source literal hit (20/20). There is no observed literal
within-partition search miss left on this prefix. The next accuracy experiment
must score semantic premise sufficiency and answer generation, not widen the
candidate pool again.

A bounded two-layer Qwen QK+OV candidate reserve was also exercised on the
first five development rows. It made six source-candidate substitutions per
question within an eight-candidate/1,024-token transient workspace, but matched
the scalar arm exactly at 3/5 literal hits and 100% source coverage while adding
substantial latency. It remains an opt-in experimental arm; the result does not
justify replacing the frozen scalar policy.

The subsequent recursive treatment uses the requested combined activation
rather than text feedback: the original question selects first-round evidence,
then `question + selected evidence` is re-encoded as a bounded Qwen window and
its QK/OV state searches a fresh lower-ranked pool. On the first five rows it
again matched 3/5 literal hits and 100% source coverage, with 2.4 additional
mean context tokens. It selected 30 genuinely new candidates through the
combined activation while remaining below eight candidates / 936 tokens per
forward. This validates the live recurrence but still cannot be promoted from
literal metrics that are already saturated.

## One-million-token context reduction

The first one-million-token construction combines ten complete locked
development histories into a single 1,039,203-token, 5,400-turn memory. This
is a deliberately harsher candidate-competition test than the isolated
LongMemEval protocol. It makes no provider calls and does not measure answer
generation.

The policy originally frozen for the 100k experiment did not scale: it reached
only 81.3% mean evidence-source coverage and all required sources on 7/10
questions. Widening source activation to 80 and its candidate pool to 1,000
raised those values to 98.3% and 9/10. Subsequent bounded sweeps found that 65
activated candidates and a 750-item pool retain the same aggregate metrics.
At an unmodified 3,250-token evidence cap that arm returns 3,176 mean tokens.

A deterministic query-aware sentence packer now runs only after durable chunks
have been retrieved. It uses the existing pySBD boundaries and lexical tokens,
keeps up to two best matching sentences in source order, preserves each durable
chunk ID, and falls back to the full chunk when no lexical match exists. It
does not retain activations, add a language-model call, or change retrieval
scores. The feature remains opt-in.

| 1M development arm (n=10) | Literal recall | Best token-F1 | Mean context | Mean source coverage | All sources |
| --- | ---: | ---: | ---: | ---: | ---: |
| Wide coverage, `a80/p1000`, 6,750 cap | 40% | 0.145 | 6,203 | 98.3% | 90% |
| Compact, `a65/p750`, 3,250 cap | 40% | 0.145 | 3,176 | 98.3% | 90% |
| Sentence-packed, `a65/p750`, 3,250 cap | 40% | 0.182 | 3,148 | 98.3% | 90% |
| **Sentence-packed, `a65/p750`, 2,250 cap** | **40%** | **0.182** | **2,178** | **98.3%** | **90%** |
| Sentence-packed, `a65/p750`, 2,125 cap | 30% | 0.173 | 2,056 | 98.3% | 90% |
| Sentence-packed, `a65/p750`, 1,500 cap | 20% | 0.158 | 1,448 | 93.3% | 80% |

The selected development point returns about 0.210% of the stored transcript,
or roughly 477:1 context compression. It cuts returned context by 64.9% from
the wide coverage arm and by 31.4% from the verified compact unpruned arm while
preserving their aggregate source and literal checks. The 2,125 result locates
the stricter literal-recall knee, so 2,250 is selected with a small margin.

These results do not establish 95% answer accuracy. They show that deterministic
linguistic preprocessing can improve the density of already-retrieved evidence;
it cannot replace semantic source routing, temporal composition, or the
downstream responder. The bound artifact and exact policy are recorded in
`data/longmemeval-million-context-pareto-development-v1.json`. A larger locked
development run is required before this becomes the default policy.

### TF-ISF, lazy source contraction, and information-rate pruning

The next source-routing pass added two bounded, opt-in channels. TF-ISF scores
the query against aggregate source/session term statistics without copying
source text. A lazy HSC-style contraction tree derives source centroids from
authoritative chunk embeddings and stores only transient parent/child pointers
and centroids. Reserving four final chunk slots for query-seeded HSC siblings
recovered the remaining expected source: mean source coverage reached 100% and
all expected sources were packed for 10/10 questions. An eight-chunk HSC reserve
was worse because it displaced a previously covered source; channel allocation,
not raw candidate volume, was the constraint.

An initial global information-gain reranker repeated that displacement failure.
It was replaced by a monotone rate-distortion filter that preserves retrieval
order and rejects only candidates below a conditional information-per-token
floor. Candidate-set IDF supplies surprise; lexical/dense score supplies query
relevance; already accepted concepts, sources, and numbers supply diminishing
returns. Multi-fact queries (sets, orderings, comparisons, all/each) receive a
0.70 threshold multiplier because repeated topical language can carry distinct
required facts.

| HSC4 1M development arm (n=10) | Literal recall | Best token-F1 | Mean context | Mean source coverage | All sources |
| --- | ---: | ---: | ---: | ---: | ---: |
| Unfiltered | 40% | 0.182 | 2,179 | 100% | 100% |
| Information floor 0.0055, no multi-fact guard | 40% | 0.182 | 2,155 | 100% | 100% |
| **Information floor 0.008, multi-fact guard** | **40%** | **0.182** | **1,986** | **100%** | **100%** |
| Information floor 0.00825, multi-fact guard | 30% | 0.177 | 1,957 | 100% | 100% |

The selected point removes 193 mean tokens (8.9%) from the HSC4 prompt without
changing any measured aggregate recall metric, returning about 0.191% of the
1,039,203-token transcript—roughly 523:1 compression. The 0.00825 boundary is
important: source coverage remained 100% while literal recall fell, because a
session-level source label cannot prove that every answer-bearing fact within
that session survived. Exact policy, hashes, and the rejected boundary are in
`data/longmemeval-million-context-information-rate-development-v2.json`.

This remains a ten-question locked-development result. No responder or judge
was called, and no 95% semantic-answer claim is made.
