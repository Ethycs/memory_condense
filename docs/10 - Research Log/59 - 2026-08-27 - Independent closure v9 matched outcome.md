# Independent closure v9 does not improve the matched control

**Status:** complete, sealed, replayed, and rejected for positive-only
composition. Against the common-renderer `S0_CONTROL_V2` result of 53/100,
both independent closure arms scored 52/100. The >=95% target remains unmet.

This entry closes the representative-bridge and artifact-global experiment
started in Research Log 49 and repaired through Research Logs 54--58. The
runtime retrieval was gold-blind. Desired-target attribution was loaded only
after retrieval artifacts were sealed and was marked `runtime_use_forbidden`.
The answer plane used Terra, and a separate Sol judge saw question, reference,
and sealed prediction. Unchanged descendant predictions inherited the already
sealed S0-v2 verdict; only changed predictions received a fresh judge call.

## Outcome at a glance

| Arm | Terra answer calls | Changed predictions | Fresh Sol calls | Inherited verdicts | Correct | Exact | Mean F1 | Paired net | Decision |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| `S0_CONTROL_V2` | 100 | -- | 100 | 0 | **53/100** | 27 | 0.410760 | -- | matched control |
| `S0_PLUS_REPRESENTATIVE_BRIDGE` | 79 | 25 | 25 | 75 | **52/100** | 26 | 0.411434 | -1 | reject |
| `S0_PLUS_ARTIFACT_GLOBAL` | 79 | 21 | 21 | 79 | **52/100** | 25 | 0.405089 | -1 | reject |

The two descendants added 158 Terra and 46 Sol calls to the already sealed
100-Terra/100-Sol parent lineage. Retrieval generation, target analysis,
ledger construction, and every replay made zero provider calls. Both arms
fail the preregistered strictly-positive paired-marginal gate and therefore do
not enter `S0_PLUS_ACCEPTED_COMPOSITION`.

## Retrieval, merge, and attribution seals

The artifact roots are:

```text
eval_results/longmemeval-1m-locked-retrieval-mechanism-arms-20260826/independent-closure-v9
eval_results/longmemeval-1m-locked-retrieval-mechanism-arms-20260826/matched-eval-spine-v2/s0-control-v2
eval_results/longmemeval-1m-locked-retrieval-mechanism-arms-20260826/matched-eval-spine-v2/s0-plus-representative-bridge
eval_results/longmemeval-1m-locked-retrieval-mechanism-arms-20260826/matched-eval-spine-v2/s0-plus-artifact-global
```

All 79 noncontiguous eligible question artifacts and all ten shard indexes
passed their sidecar, question, source, S0, route, packet, and receipt checks.
The merged generator preserves those 79 artifacts and the matched projection
expands them to 100 rows by emitting exact no-op parent fallbacks for the 21
ineligible questions.

| Artifact or identity | SHA-256 |
| --- | --- |
| frozen base retrieval | `e36b54ec6171aa7b40f75682ad85e5822a64d45bc411ffe03bcd9cad0222007f` |
| v9 eligibility manifest file | `748bd56a7efb8fd70d36bc96f099a53fc506469565577de9635908f6773bdee1` |
| v9 retrieval preflight file | `268cb5bfa70661de470b5142163d9447a199c05ce713233cb75ff7ce25ec4451` |
| closure policy receipt | `21e1b521247eb16fd1ac1f8ac3252f86e61930fa078f4b2e47800335a8d512d1` |
| merged v9 retrieval generation file | `cf541c40f0749dcf9e436080c56dcf251232fd9ac7c844be49e2dfd8764a7ee5` |
| target-owner plan file | `b96786a4ef87a2958e385939b31857e06a33a1bd1577eb693e6a4a409f8356ff` |
| target-owner plan identity | `2cabfbb103929c68dea47368502875444903ced282c708cba45ef26bee14d888` |
| target-analysis file | `496eb1d9a9e3b47f152b1805539f72702d63cd542534ecc80e81878bf6bebe46` |
| target-analysis self-seal | `ba2a971b98bcfa9f35ab503839abf89f09b377163ae2377829b2f1d015f71043` |
| matched population identity | `9b8ad9337cfece1306358d0e03682a977f1b289a14b6ff7bfe40c90e6e2cb246` |
| representative structural projection identity | `90d8faef2305c5a38f91ef3286fe8d8f07b2179d4696c764750260bbe1fdcfc9` |
| global structural projection identity | `d53c02be3c48c180d2fbbb3cf204378a019b4f9037900045841f94db55e084e3` |

The complete retrieval shard ledger is:

| Offset | Eligible questions | Shard SHA-256 |
| ---: | ---: | --- |
| 0 | 7 | `77133620564510efbc2554e29fb3a587c9de9b6b02998fe803fbb1d80bd66b36` |
| 10 | 7 | `324ef35e4d835d6b0e1dedae82fc1d6ae63be1e9f99e9f3968fa74dbe9517873` |
| 20 | 8 | `39ee51a5372735a0d98a0637b8e095157d94c664c65729661f5453fbeb31097d` |
| 30 | 7 | `62c56a62a141de5103b0f72275103ced54242c4874364038861c9188d1110eeb` |
| 40 | 8 | `3eb1f391debc2fc8871037a40d07e22a6db0d792c80beec5a8fee97bd4ae4c31` |
| 50 | 8 | `723ccc8af286a6e8c5872b8a7d67ce103a183bf316e9aa8cc8d5a9fb2b540883` |
| 60 | 8 | `d0f9b3611031c6f40de2f139476cd88ba870582ed8440dbc52c7e489af14f166` |
| 70 | 9 | `6a7f4d4d5fcf0f7c91daa654e361b3d05f062184de5e5002d1517af219a2a9d7` |
| 80 | 8 | `743b45386aba01db8e7c41f5c0b9f309238f407e4de13f3e35ebb52b94d39c75` |
| 90 | 9 | `d9395c3327df24bf7fce05ac73cfdc1c2476369c7f1d91095782288bfbdb8911` |
| **total** | **79/79** | -- |

## Answer, judge, and score seals

For each run/replay pair below, the two files have the same SHA-256. The
runtime ledger and score ledger pairs are also byte-identical.

| Plane | `S0_CONTROL_V2` | Representative bridge | Artifact global |
| --- | --- | --- | --- |
| answer preflight | `96c109c64fbf6232e4cfa3fbc252aa8a008624d1e1bffe29ddbf0222d8f6e315` | `83e2b13873b90c1e62eb7da4e995833ed4b32bb0e36b689246e0d3a19253e077` | `f65be68a27314d6f254ffbb20ed8dc0e975e5c449f31d416c585912b9781efb5` |
| answer run / replay | `1a2545655d4a5e2061dc1b80efae39c7f8c70f5dc394f36c97d1312f70f39d8a` | `008d1ad2cdc34392feec696aa4fa61b0644b1b85f56ff8ec24c711b360fbf311` | `834f2e26852ea31a94a5fd2d72251ad102322283f8742a82b28c9a80ee1e8d84` |
| runtime ledger / replay | `f4f6d1a52ceea2b7f65cb66f51bb4925c1db9d20253c7ada7167216285a7d45b` | `4a4af8f121c087a5068f52fb40cc785cfa5ba18cce5452c8c8aebf0b4eabe3dc` | `6aa8c837a33aef2e91c01f3ee7a02ad1d9c1c2df10e6451e5af9b5ec72864e20` |
| judge preflight | `5ad11d9742cfe1de841c75106c6b434d480280f431d505195ed7c1753bc890d1` | `64619e790f4e6082a2f5b05dc0cdaa67a8e74e858f8aeabb5f70472bda9ef617` | `3bb760b469feb74ca2e2c6bfc2709aa363e98569cd320b7378dbe60baaa6e027` |
| Sol judge run / replay | `05fec9a7f284bb4e95d286f44e7378a8bbc1737a03e7c2ed60aefd50e6ddc689` | `bc95b3ea56caeb7eed1c2729794421ff21c05eb586b0318bbab53000b54be23b` | `bacbbd586bd3bfb20a6f3835604ceb5da22c1a95ec7f9c4bd6e73e7840741693` |
| score ledger / replay | `3422ce2825bdcdc347c8307bd3fed5a46de3dff6d33510c8bc3a3ba1c31c56e1` | `9ec4b7ca857bcd993661432bb7519cef25f3360821340b8738bfc82235e3e17b` | `54469adb5c13bd842b58107a54d3ee385334fba9ed3d29c14d3b26e300533340` |

Every answer and judge artifact records zero retained request-token-state
bytes. Completion provenance records zero retained transformer-token-state
bytes and `persisted_transformer_token_state == false`. Replay used no
provider client and reproduced the sealed answer, runtime, judge, and score
bytes exactly. The descendant runtime ledgers contain 200 rows apiece: one
closure-stage row and one answer-observation row for every question. They
record 79 provider calls, zero local calls, zero historical calls, and
`gold_loaded == false`.

Provider-native token totals were not returned by this SDK route, so their
recorded totals are zero and completeness flags are false. The sealed token
proxy is the budget authority; this is an accounting limitation, not a seal
or replay failure.

## Protected budgets and actual use

| Plane | Declared cap | Representative actual | Global actual |
| --- | ---: | ---: | ---: |
| retrieval addition per question | 2,048 tokens | max 1,145 | max 1,196 |
| final answer prompt per called question | 8,000 tokens | max 6,624 | max 6,426 |
| answer output | 256 tokens | 359 aggregate completion-token proxy | 337 aggregate completion-token proxy |
| changed-only judge prompt | 8,000 tokens | max 218; 3,628 aggregate | max 216; 3,000 aggregate |
| judge output | 1,024 tokens | 530 aggregate completion-token proxy | 445 aggregate completion-token proxy |

For reference, the matched S0-v2 parent used a maximum 5,525-token answer
prompt and a maximum 234-token judge prompt. Its aggregate answer/judge prompt
proxies were 449,292 and 14,026, respectively. The descendant aggregate
answer prompt proxies were 439,728 and 417,733 because only 79 eligible rows
made new answer calls; 21 rows reused the sealed parent prediction.

The final operational membership partition was exact:

| Arm | Raw candidates | Selected before S0 dedup | Exact-S0 exclusions | Final-repack not admitted | Admitted | Stage dispositions |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| representative bridge | 3,712 | 444 | 0 | 0 | 444 | 79 added, 21 no-op |
| artifact global | 18,246 | 712 | 218 | 1 | 493 | 79 added, 21 no-op |
| **union** | **21,958** | **1,156** | **218** | **1** | **937** | -- |

Deduplication occurred after selection. In particular, the 218 global
exclusions remain credited as discovered/selected and are then removed by
exact protected-S0 identity before admission. The single additional global
row is the exact final-repack `not_admitted` partition; no row disappears
between these accounting stages.

## Target reach, selection, and admission

The target analysis is posthoc and provider-free. Its universe is the frozen
263-target registry: 188 source targets, 71 relation targets, and four
coverage checks. Relation-operand completeness is diagnostic and does not
count as a formal relation hit.

| Route and stage | Events | Formal targets | Source targets | Closure-owner targets | Complete relation operands | Alternate-owner rescues |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| representative raw reach | 3,712 | 0/263 | 0/188 | 0/28 | 0/71 | 0 |
| representative selected | 444 | 0/263 | 0/188 | 0/28 | 0/71 | 0 |
| representative admitted | 444 | 0/263 | 0/188 | 0/28 | 0/71 | 0 |
| global raw reach | 18,246 | 136/263 | 135/188 | 11/23 | 49/71 | 24 |
| global selected | 712 | 65/263 | 65/188 | 2/23 | 17/71 | 13 |
| global admitted | 493 | 46/263 | 46/188 | 2/23 | 8/71 | 8 |
| **union raw reach** | **21,958** | **136/263** | **135/188** | **35/51** | **49/71** | **24** |
| **union selected** | **1,156** | **65/263** | **65/188** | **15/51** | **17/71** | **13** |
| **union admitted** | **937** | **46/263** | **46/188** | **10/51** | **8/71** | **8** |

The surprising result is structural, not merely a prompt-packing loss. The
representative route produced many atoms but zero hits under the common
external target identity. The global route recovered 24
representative-owned targets at raw reach, 13 after selection, and eight
after admission. Meanwhile, only 2/23 global-owned targets survived its own
selection and admission. The route labels therefore do not align well with
the memory responsibilities assigned to them, and most raw target reach is
lost before the final answer packet.

## Rescues, regressions, and categories

| Arm | Rescues | Regressions | Net |
| --- | --- | --- | ---: |
| representative bridge | ordinal 14 (`multi-session`); ordinal 70 (`single-session-assistant`) | ordinals 29 and 48 (`temporal-reasoning`); ordinal 87 (`multi-session`) | -1 |
| artifact global | ordinal 5 (`single-session-preference`); ordinal 14 (`multi-session`) | ordinal 21 (`temporal-reasoning`); ordinals 72 and 87 (`multi-session`) | -1 |

| Category | Questions | S0-v2 correct | Representative correct (delta) | Global correct (delta) |
| --- | ---: | ---: | ---: | ---: |
| knowledge-update | 16 | 15 | 15 (0) | 15 (0) |
| multi-session | 27 | 11 | 11 (0) | 10 (-1) |
| single-session-assistant | 11 | 4 | 5 (+1) | 4 (0) |
| single-session-preference | 6 | 0 | 0 (0) | 1 (+1) |
| single-session-user | 14 | 10 | 10 (0) | 10 (0) |
| temporal-reasoning | 26 | 13 | 11 (-2) | 12 (-1) |
| **all** | **100** | **53** | **52 (-1)** | **52 (-1)** |

### Ordinal 72 is judge noise, not a silent correction

The parent prediction was `I don't know`; the global descendant was `I don't
know.`. Exact bytes changed only by punctuation, so the changed-only protocol
correctly sent the descendant to Sol. The sealed parent judge had called the
first answer correct because the chili-pepper count was absent. The fresh
global judge called the punctuation-equivalent answer incorrect because it
omitted the known count of five tomato plants. The representative descendant,
`5 tomato plants; I don't know how many chili pepper plants.`, was judged
correct.

This is a real inconsistency between independent judge calls. The
authoritative sealed aggregate remains 52/100; no verdict was edited after
inspection. Even a hypothetical same-verdict treatment for ordinal 72 would
move global only to 53/100, a zero marginal that still fails the strictly
positive composition gate.

## Decision and claim boundary

Neither closure arm improves the matched parent. Both whole arms are rejected
from positive-only composition. The isolated category gains are posthoc and
do not authorize cherry-picking a question/category oracle into runtime.
Future work should repair the route-to-target identity mismatch and the steep
raw-reach-to-admission loss before spending another full answer/judge budget.

The often-cited **84/100 was never an observed score**. It was the earlier
counterfactual ceiling `56 + 28`: what the fixed-S1 lineage would have reached
if every diagnosed nominal-full-source failure were repaired with zero
regression. No responder or composed memory system produced or replayed
84/100. The observed matched results in this campaign are 53/100 for S0-v2
and 52/100 for each closure descendant. There is no 95/100 claim; the formal
target remains unpassed and this locked population is analysis-used.
