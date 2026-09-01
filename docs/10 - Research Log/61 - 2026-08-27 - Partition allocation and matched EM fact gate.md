# Partition allocation closes selection loss; the matched EM fact gate adds one point

**Status:** two isolated locked-100 diagnostics are complete and sealed. The
provider-free partition-scan v2 repair preserves v1's 19/27 missing-source
candidate reach while raising selection and admission from 14/27 to 19/27.
It has no answer-accuracy result. Separately, a parent-preserving EM fact gate
scores **54/100** against the common-renderer S0-v2 parent's **53/100**: one
rescue, zero regressions. No >=95% result follows.

These are not stages of one measured stack. The fact gate reuses the sealed
fixed-S1 EM representation and does not consume partition-scan v2 evidence.
Their numbers therefore cannot be added.

## Result summary

| Experiment | Structural or answer result | Calls in this experiment | Decision |
| --- | --- | ---: | --- |
| partition scan v1 | candidate 19/27; selected 14/27; admitted 14/27 | 0 | selection budget left five reachable targets behind |
| partition scan v2 | candidate **19/27**; selected **19/27**; admitted **19/27** | 0 | retain as the stronger construction adapter; answer value remains unmeasured |
| matched S0-v2 parent | **53/100** semantic | already sealed | matched parent |
| routed EM fact gate | **54/100** semantic; +1 rescue, zero regressions | 25 Terra answers + 14 Sol judgments | retain as a conservative positive answer-time cell |

The two result types have different denominators and meanings. The 19/27
figure is exact desired-source identity among the eligible sources missed by
S0 and both raw closure pools. The 54/100 figure is independent semantic
answer accuracy across the complete locked population.

## Partition scan v2

### What changed from v1

V1 completely scanned the same four gold-blind routed partitions but selected
globally from one exact query-centred span per source. Five desired sources
were constructible yet lost at selection. V2 leaves the coarse router
unchanged and modifies only within-scope reduction and allocation:

1. retain the two best exact query-centred spans per source, capped at 48
   tokens per span;
2. allocate the 2,048-token budget across the four ranked partitions with
   weights `4:2:1:1`;
3. reserve `24/25` of each allocation for first-span source coverage;
4. spend the remainder on second spans; and
5. exclude exact protected-S0 overlaps only after selection.

The runtime inputs remain the dated question, protected S0, and frozen lexical
index. It searches the full ten-history combined store, never filters by a
question-ID or source prefix, opens no reference answer, and makes no provider
call. The 21 question-only-ineligible rows are exact no-ops; 79 rows perform a
complete scan of their selected partitions.

### Full-population accounting

| Measure | v1 | v2 |
| --- | ---: | ---: |
| scanned content rows | 237,539 | 237,539 |
| candidate source memberships | 15,202 | 15,202 |
| candidate exact spans | 15,202 | 30,371 |
| selected source memberships | 7,472 | 8,309 |
| selected exact spans | 7,472 | 9,371 |
| selected second spans / multi-span memberships | 0 | 1,062 |
| exact protected-S0 exclusions | 259 | 378 |
| admitted source memberships | 7,213 | 8,079 |
| admitted exact spans | 7,213 | 8,993 |
| maximum selected tokens before dedup | 2,048 | 2,048 |

V2's mean admitted use over the 79 eligible questions is 1,945.57 tokens; its
maximum admitted use after S0 dedup is 2,045. The larger span count is
intentional representation breadth, not new source construction: candidate
source membership is exactly unchanged.

### Corrected 27-source funnel

| Stage | v1 | v2 | Delta |
| --- | ---: | ---: | ---: |
| correct source among reduced candidates | 19/27 | 19/27 | 0 |
| correct source selected before S0 dedup | 14/27 | **19/27** | +5 |
| correct source admitted after S0 dedup | 14/27 | **19/27** | +5 |

The five source rescues occur at ordinals 7, 76, 77, and twice at 86; there
are no v1-to-v2 target regressions. Of the 19 admitted desired sources, ten
receive two exact spans and nine receive one. Eighteen of the 19 have nonzero
question-token overlap in the chosen span proxy. The ordinal-7 rescue does
not, which is a concrete warning against equating source identity with useful
answer evidence.

The remaining eight sources were never in the four scanned partitions:
ordinal 43 contributes two, ordinal 54 one, ordinal 61 three, and ordinal 93
two. Allocation cannot recover them. They require a broader or better coarse
router, not another within-partition packing change.

### Construction claim boundary

The target registry was opened only after the runtime generation and its
dependencies had verified. It supplies desired source IDs, not gold character
spans. Consequently, exact source reach plus exact-span provenance proves
that authentic text from the desired history was selected. It does **not**
prove that the excerpt contains the answer-bearing fact, that a compressor
will preserve it, or that a final responder will answer correctly. V2 has no
Terra answer or Sol judge result and contributes zero points to the semantic
score.

## Matched routed EM fact gate

### Parent-preserving design

The fact-gate experiment answers a different question: can a conservative
facts-only representation improve the common-renderer S0-v2 result without
rebuilding retrieval or exposing every question to a new answer call?

The separately pinned route policy classifies from dated question text only
and admits `numeric_reduce` and `state_chain`. It then validates the already
selected fixed-S1 EM delta and its already produced exact-cited fact packet.
Selection precedes exact protected-S0 deduplication. The final answer prompt
contains cited facts and no raw EM rows. A denied route, invalid or empty fact
packet, non-novel delta, or prompt overflow returns the exact sealed parent
prediction.

This gate explicitly records `construction_recall_claimed=false` and
`source_target_expansion_claimed=false`. It changes representation and the
answer operator; it does not retrieve a missing source.

### Gate and prompt accounting

The question-only classifier assigns 32 questions to `numeric_reduce`, nine
to `state_chain`, and 59 to the four denied routes. Of the 41 route-admitted
questions, 25 compile valid fact packets: 20 numeric and five state. Fifteen
fall back for empty or invalid cited facts and one for invalid compression.
Together with the 59 route denials, this yields 75 exact parent fallbacks.

Across the 25 compiled rows:

| Representation stage | Count |
| --- | ---: |
| selected evidence identities before S0 dedup | 1,229 |
| exact protected-S0 exclusions | 808 |
| admitted EM-delta identities | 421 |
| cited facts | 41 |
| unique cited evidence identities | 40 |
| unique cited sources | 36 |
| raw EM rows in final prompts | **0** |

The Terra answer budget is 8,000 input tokens and 256 output tokens per call.
Observed answer prompts total 88,463 proxy tokens across 25 calls, with a
maximum of 3,889; output totals 58 proxy tokens. The 25 calls produce 14
predictions whose bytes differ from the parent and 11 that are unchanged.

The changed-only Sol judge then receives question, reference, and sealed
prediction for exactly those 14 changed rows. The other 86 verdicts inherit
the sealed parent judgment. Its budget is 8,000 input and 1,024 output tokens;
observed judge prompts total 1,809 proxy tokens with a maximum of 144, and
output totals 268 proxy tokens.

### Authoritative semantic outcome

| Metric | S0-v2 parent | EM fact gate | Paired delta |
| --- | ---: | ---: | ---: |
| independent semantic correct | 53/100 | **54/100** | **+1** |
| normalized exact match | 27/100 | 31/100 | +4 |
| mean normalized F1 | 0.410760 | 0.434760 | +0.024000 |
| rescues | -- | 1 | +1 |
| regressions | -- | 0 | 0 |

The sole semantic rescue is ordinal 35, a `numeric_reduce` age-difference
question. The parent was incorrect; the gated prediction `43` was judged
correct. None of the 53 parent-correct rows regressed. This validates exact
fallback as a regression guard for this sealed sample, but the +1 marginal is
far too small to support a path-to-95 claim by itself.

### Replay and state boundary

Answer, runtime-ledger, judge, and score replays are byte-identical to their
live artifacts. The answer replay uses 25 checkpoint hits and zero calls; the
judge replay uses 14 hits and zero calls. Runtime accounting records 200 rows,
25 new provider calls, zero local-model calls, and `gold_loaded=false` before
the posthoc judge plane. Both providers record zero retained request or
transformer token-state bytes.

The SDK did not return native prompt/completion token totals, so provider
token-completeness flags are false. The pinned `cl100k_base` prompt proxy is
the enforceable budget authority; this is an accounting limitation, not a
prompt-cap or replay failure.

## Exact seals

### Shared and partition-scan v2

| Artifact or identity | SHA-256 |
| --- | --- |
| frozen base retrieval | `e36b54ec6171aa7b40f75682ad85e5822a64d45bc411ffe03bcd9cad0222007f` |
| matched population identity | `9b8ad9337cfece1306358d0e03682a977f1b289a14b6ff7bfe40c90e6e2cb246` |
| eligibility manifest | `748bd56a7efb8fd70d36bc96f099a53fc506469565577de9635908f6773bdee1` |
| partition-scan v1 generation | `48c9f0b5eb2eb8f49a47002ce0beed843bbb6b478b45bf311d5c8d6c6e34f3f4` |
| partition-scan v1 missing-source analysis | `01248bc78a1721951cc1131f36707516701bbbe5a50481f6a75f930e196670df` |
| partition-scan v2 generation | `671f0a3418364f544e61897c42569407805e827ae558980760289dae6b5cf388` |
| partition-scan v2 missing-source analysis | `16e6c8555efe8bb7a6c4691f76caeb131422aed2d1846f204e3afb8eadaaca42` |

The final v2 artifacts are under:

```text
eval_results/longmemeval-1m-locked-retrieval-mechanism-arms-20260826/partition-scan-v2-r96
```

The earlier `partition-scan-v2` directory is a superseded 90%-reserve pilot;
`partition-scan-v2-r96` is the authoritative 24/25-reserve result.

### Matched EM fact gate

| Artifact or identity | SHA-256 |
| --- | --- |
| route policy | `97d353def0d81183419e631b3227a8b7221c1e2d8acc4cb932486d95704a89c6` |
| source EM compression | `4e4665845c5e7df6af779b599d3fb97a010041bdb893b7763ef84e678c868393` |
| source EM run / replay | `af2ee321cbd4d624b753ac942072bbe2fd54d49b86384ae7fdb13d6b46cc3db9` |
| parent S0-v2 answer run | `1a2545655d4a5e2061dc1b80efae39c7f8c70f5dc394f36c97d1312f70f39d8a` |
| parent S0-v2 runtime ledger | `f4f6d1a52ceea2b7f65cb66f51bb4925c1db9d20253c7ada7167216285a7d45b` |
| parent S0-v2 Sol judge | `05fec9a7f284bb4e95d286f44e7378a8bbc1737a03e7c2ed60aefd50e6ddc689` |
| parent S0-v2 score ledger | `3422ce2825bdcdc347c8307bd3fed5a46de3dff6d33510c8bc3a3ba1c31c56e1` |
| answer preflight | `01387426e146b36744779595e8f97210f444d36d051ac69570ede5af3ec042e7` |
| answer prompt population | `d705e2e2745c652489998938566f895148185e602b5c12aa9f8eb2b913dd6877` |
| answer run / replay | `463c05a32f7b4625e454292b5767f61a77758a4a876122502bd74fb8267e7bd0` |
| runtime ledger / replay | `064a31fb1ab61b11b102aa195378def4e732f1a2aede63ef169c9bfa29159b06` |
| change projection identity | `ddbaafc2c919287ae6193fbc9a677002815aa800b0ef6cff71256cf7fd682095` |
| changed-only judge preflight | `3a2a7fdc5245b4528adc187e08ddb6e3fcfef91b171dbc198e3f38aae708eb92` |
| Sol judge / replay | `b33aeaed4527e36622b211254e3f25a698d24e8f64fb8e72a9e8898e43f20712` |
| score ledger / replay | `5d01f7993f9c8ce15287f9c876caa40323ba8c5c238ee3ea6902f75189c2002e` |

The fact-gate artifacts are under:

```text
eval_results/longmemeval-1m-locked-retrieval-mechanism-arms-20260826/matched-eval-spine-v2/s0-plus-routed-em-fact-gate-v1
```

## Decision and next measurement

Partition allocation v2 is a real structural improvement over v1, but it
exhausts only the within-top-four selection defect. A broader gold-blind
construction method must address the eight coarse-routing misses. Before any
new evidence becomes a candidate for composition, its selected excerpts need
answer-bearing validation through a separately budgeted representation and
answer path.

The EM fact gate is safe enough to retain as one isolated positive cell on
this analysis-used population. Its exact fallback behavior prevents measured
regression here; it does not establish general zero-regression behavior. The
next composed run must keep construction, representation, answering, and
judging budgets separate, use only preregistered positive cells, and confirm
on a fresh locked population. The >=95% gate and fair Mem0 comparison remain
open.
