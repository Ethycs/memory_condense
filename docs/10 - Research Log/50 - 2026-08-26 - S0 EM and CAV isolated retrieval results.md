# S0, EM facts, and genuine CAV links now have isolated locked results

**Status:** complete and byte-replayed for the first three arms of the matched
retrieval-mechanism matrix. On the analysis-used locked 100-question
population, `S0_CONTROL` scored **57/100**, `S0_PLUS_EM_FACTS` scored
**60/100**, and `S0_PLUS_CAV_LINKS` scored **53/100**. EM rescued eight
questions and regressed five for a net gain of three. Genuine CAV linking
rescued two and regressed six for a net loss of four, so it is excluded from
positive-only composition.

This entry records the completed portion of the experiment specified in
[Research Log 49](49%20-%202026-08-26%20-%20Matched%20retrieval%20mechanism%20matrix%20roadmap.md).
These are analysis results from a population that has already been inspected;
they are **not** a held-out confirmation and they do **not** establish the
95/100 objective. The best measured arm here is 60/100, not 95/100.

## Isolated comparison boundary

Both treatment arms are direct children of the exact same S0 control; CAV is
not stacked on EM in this comparison:

```text
S0_CONTROL                         57/100
├── S0_PLUS_EM_FACTS               60/100  (+8 rescue, -5 regress, net +3)
└── S0_PLUS_CAV_LINKS              53/100  (+2 rescue, -6 regress, net -4)
```

EM represents the fully selected `S1 - S0` episodic delta as cited atomic
facts after selection and deduplication. CAV exposes genuine two-pass concept
links over S0 while keeping the raw S0 evidence packet fixed. Thus the EM arm
measures an episodic selection-plus-representation intervention, whereas the
CAV arm measures a linking intervention with no membership change.

All answer calls used Terra and all semantic judgments used Sol with zero
retries. Gold was unavailable to retrieval, compression, feature generation,
and answering. The judge opened the pinned question and reference only after
the answer artifact and its zero-call replay verified. Descendant judges sent
only changed predictions to Sol and reused the sealed S0 verdict for every
byte-identical unchanged prediction.

| Arm | Pre-answer work | New judgments / Sol calls | Terra answer calls | Correct | Rescue / regress | Net | Composition gate |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| `S0_CONTROL` | none | 100 / 100 | 100 | 57/100 | -- | -- | control |
| `S0_PLUS_EM_FACTS` | 100 Terra compression calls | 43 / 43 | 62 | 60/100 | 8 / 5 | **+3** | positive overall |
| `S0_PLUS_CAV_LINKS` | 4 local Qwen encoder batches | 31 / 31 | 100 | 53/100 | 2 / 6 | **-4** | **exclude** |

The physical call populations above are exact, not upper bounds. The EM arm
used 162 Terra calls in total: 100 compression calls followed by 62 dependent
answer calls. The CAV feature phase used local Qwen execution and made zero
provider calls; its provider-bearing phases were exactly 100 Terra answers and
31 changed-only Sol judgments.

## EM facts: positive overall, concentrated by demand

All 100 selected episodic deltas received one compression attempt. Sixty-two
compressions were valid, 35 were empty, and three were invalid or ungrounded.
Only the 62 valid fact packets received an answer call. The other 38 questions
fell back to the sealed S0 prediction. Of the 62 dependent answers, 43 changed
the prediction, so 43 new Sol calls were necessary and the other 57 S0
verdicts were reused.

The question-only demand slices show that the global gain is not uniform:

| Question-only demand | Questions | Changed / Sol | S0 -> EM correct | Rescue / regress | Net |
| --- | ---: | ---: | ---: | ---: | ---: |
| `direct_extract` | 24 | 9 | 16 -> 15 | 2 / 3 | -1 |
| `numeric_reduce` | 32 | 12 | 17 -> 20 | 4 / 1 | **+3** |
| `set_join` | 1 | 1 | 1 -> 1 | 0 / 0 | 0 |
| `state_chain` | 9 | 3 | 8 -> 9 | 1 / 0 | **+1** |
| `synthesize` | 6 | 4 | 1 -> 1 | 0 / 0 | 0 |
| `temporal_timeline` | 28 | 14 | 14 -> 14 | 1 / 1 | 0 |
| **all** | **100** | **43** | **57 -> 60** | **8 / 5** | **+3** |

EM helped numeric reduction and state chains, tied three demand classes, and
hurt direct extraction. Its evidence-topology view was similarly mixed:
dispersed joins improved from 38/65 to 42/65 (five rescues, one regression,
net +4), the single local-pair question stayed 0/1, and point questions fell
from 19/34 to 18/34 (three rescues, four regressions, net -1). The arm passes
the global positive-marginal gate, but a question-demand composer should not
copy its negative direct-extraction cell.

## Genuine CAV links: non-null intervention, negative semantic marginal

The CAV feature pass freshly encoded the locked questions and evidence with
the pinned Qwen3-8B layer-0 prefix encoder. It covered 3,558 globally unique
texts--100 questions and 3,458 evidence texts--in exactly four bounded
execution chunks, each dispatched through one Qwen `encode_layers` call.
Those four calls contained 445 internal transformer forward batches, with one
model load and zero truncated rows. This was a local feature phase, not four
external provider calls.

For each question the genuine fixed two-pass router used three concepts and
the top four extraction links per concept to build a bounded guide. The old
X/X1 ordering proxy was explicitly not consumed. The run-level invariants are
`evidence_additions = 0` and `exact_s0_membership_and_order = true`: every CAV
answer saw the exact same ordered raw S0 evidence as its parent, plus only the
link guide. This is therefore a genuine linking test, not a hidden retrieval
or reranking intervention.

All 100 questions received one Terra answer attempt. Thirty-one predictions
changed and required 31 Sol calls; 69 unchanged predictions reused their S0
verdicts. The per-demand result was:

| Question-only demand | Questions | Changed / Sol | S0 -> CAV correct | Rescue / regress | Net |
| --- | ---: | ---: | ---: | ---: | ---: |
| `direct_extract` | 24 | 5 | 16 -> 15 | 0 / 1 | -1 |
| `numeric_reduce` | 32 | 8 | 17 -> 16 | 2 / 3 | -1 |
| `set_join` | 1 | 0 | 1 -> 1 | 0 / 0 | 0 |
| `state_chain` | 9 | 3 | 8 -> 7 | 0 / 1 | -1 |
| `synthesize` | 6 | 4 | 1 -> 0 | 0 / 1 | -1 |
| `temporal_timeline` | 28 | 11 | 14 -> 14 | 0 / 0 | 0 |
| **all** | **100** | **31** | **57 -> 53** | **2 / 6** | **-4** |

The posthoc evidence-topology slices were also non-positive:

| Evidence topology | Questions | Changed / Sol | S0 -> CAV correct | Rescue / regress | Net |
| --- | ---: | ---: | ---: | ---: | ---: |
| `dispersed_join` | 65 | 18 | 38 -> 37 | 2 / 3 | -1 |
| `local_pair` | 1 | 0 | 0 -> 0 | 0 / 0 | 0 |
| `point` | 34 | 13 | 19 -> 16 | 0 / 3 | -3 |
| **all** | **100** | **31** | **57 -> 53** | **2 / 6** | **-4** |

No demand class and no topology class had a strictly positive net. The
current CAV arm is therefore excluded from accepted composition. This result
does not prove that CAV links can never help; it establishes that this exact
fresh feature bank, bounded guide, responder, and locked population do not
earn a composition slot.

## Canonical artifacts and replay identities

The artifact root is
`eval_results/longmemeval-1m-locked-retrieval-mechanism-arms-20260826/`.
Run and judge replays reconstructed canonical bytes and therefore have the
same file hash as their source artifact.

| Arm | Artifact | Source SHA-256 | Replay SHA-256 |
| --- | --- | --- | --- |
| S0 | `s0-control-v1/run.json` | `a713328485ebef452a0dd30626a7ffc20126999162723cb543da4f94a87b8e68` | `a713328485ebef452a0dd30626a7ffc20126999162723cb543da4f94a87b8e68` |
| S0 | `s0-control-v1/semantic-judge-sol.json` | `1c9ea03121478edd053c666bfffb8eaf1db508f001df76367ce14adc8f5022cb` | `1c9ea03121478edd053c666bfffb8eaf1db508f001df76367ce14adc8f5022cb` |
| EM | `s0-plus-em-facts-v1/compression.json` | `4e4665845c5e7df6af779b599d3fb97a010041bdb893b7763ef84e678c868393` | no standalone replay file |
| EM | `s0-plus-em-facts-v1/run.json` | `af2ee321cbd4d624b753ac942072bbe2fd54d49b86384ae7fdb13d6b46cc3db9` | `af2ee321cbd4d624b753ac942072bbe2fd54d49b86384ae7fdb13d6b46cc3db9` |
| EM | `s0-plus-em-facts-v1/semantic-judge-sol.json` | `13913b6bc95f1dca8d5c974fdb7bcf8feae4bae44f5f7ca78d6336ce66016cf8` | `13913b6bc95f1dca8d5c974fdb7bcf8feae4bae44f5f7ca78d6336ce66016cf8` |
| CAV | `s0-plus-cav-links-v1/features.json` | `b7dc8de695dd0d298ad0bdc100fb8a195005052031f02f694af44ada824cfafa` | strictly reverified during run replay; no standalone replay file |
| CAV | `s0-plus-cav-links-v1/run.json` | `6052f52b7835848aa8e9578703c6bb131460e0eb54c8c64e70bc42dfd783ca49` | `6052f52b7835848aa8e9578703c6bb131460e0eb54c8c64e70bc42dfd783ca49` |
| CAV | `s0-plus-cav-links-v1/semantic-judge-sol.json` | `8f44f3a259b11615f3009c5ae1d047cc6b0a94290b84a3d13b915adbd82fcfb0` | `8f44f3a259b11615f3009c5ae1d047cc6b0a94290b84a3d13b915adbd82fcfb0` |

The replay columns refer respectively to `run-replay.json` and
`semantic-judge-sol-replay.json`. Replays reopened the sealed journals and
made zero provider calls.

## Immutable desired-target plan

The posthoc target-owner plan freezes a desired memory universe of 263
targets: 188 benchmark-native source targets, 71 relation targets, and four
unsupported-conclusion coverage checks. Every desired target has exactly one
primary owner, the owner sets are pairwise disjoint, their union equals the
declared universe, and the unassigned count is zero.

| Primary owner | Desired targets |
| --- | ---: |
| S0 | 68 |
| EM facts | 67 |
| Representative bridge | 28 |
| Artifact global | 23 |
| Hebbian | 6 |
| CAV links | 71 |
| **primary-owner union** | **263** |

The immutable plan is
`target-owner-plan-v1/target-plan.json`, file SHA-256
`b96786a4ef87a2958e385939b31857e06a33a1bd1577eb693e6a4a409f8356ff`.
Its internal plan identity is
`2cabfbb103929c68dea47368502875444903ced282c708cba45ef26bee14d888`,
and its desired-universe identity is
`ade97c0eb759b0b6428d10358551f3ca612fcc326f58e2e122e97b5d9a4e355f`.
The builder loaded zero answer-run or judge inputs and made zero provider
calls. The plan is analysis-only and forbidden from runtime routing or answer
prompts.

The **desired universe** and the **candidate union** are deliberately
different objects:

- The desired universe is the fixed external evaluation denominator above.
  Its targets remain present even if no mechanism discovers them.
- A candidate union is only the set of route-local targets that the tested
  mechanisms happened to propose. It can omit desired targets and can contain
  alternate or duplicate routes to the same target.
- Defining completeness from the candidate union would make recall
  tautological: an undiscovered desired target would disappear from both the
  numerator and denominator. It must never be reported as desired-universe
  coverage.
- Structural mechanism ledgers record discovery before post-selection dedup
  and admission after dedup/budgeting. The posthoc registry joins those events
  to the immutable owner plan; runtime ledgers do not assign semantic primary
  ownership. Discovery credit survives dedup even when the final packet omits
  an exact S0 duplicate.

The plan file was frozen during this analysis campaign, after the population
was already analysis-used. Its deterministic, answer-independent construction
prevents outcome-conditioned reassignment for the remaining arms, but it does
not turn this population into a held-out confirmation set.

## Decision and remaining claim boundary

The isolated apparatus is now doing the intended causal comparison: add one
mechanism to S0, preserve its protected parent, judge only changed outputs, and
admit only positive semantic marginals. EM facts earn a positive overall
result, although their useful effect is concentrated. Genuine CAV links fail
the gate and must not be included merely because they are a more complex
layer.

The next composition must preserve S0 wherever an isolated mechanism does not
have a strictly positive preregistered cell. Any composed score on this locked
100 remains a development/analysis result. A 95/100 claim requires freezing
the final policy and reproducing it on an untouched confirmation population
with the same answer and independent-judge boundaries.
