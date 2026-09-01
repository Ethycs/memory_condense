# A1a raw-retained paired prompt boundary

Date: 2026-08-30

## Result

The provider-free A1a treatment boundary is implemented, sealed, and
byte-replayed over the accepted R7 A1 classifier dispositions. It constructs
**11 raw-retained treatment requests** from exactly relevant plus unresolved
leaves and **11 renderer-matched fixed-union control requests**. Only an
explicit `definitely_irrelevant` disposition is excluded, and exclusion occurs
after the exact 381-leaf union is fixed.

The classifier seal contains 39 relevant, 305 definitely irrelevant, and 37
unresolved decisions. A1a therefore retains **76/381 leaves**. The runtime and
its replay have artifact SHA-256
`94be872da1f527927d20cf1b78de1d7c704ee6117e5976848a0b7239ee8e0ae2`.
No provider call, fact compilation, terminal answer, or model-state retention
occurred in A1a construction.

The separate target-informed audit is **NO-GO**: **25/26 semantic atoms** and
**28/29 distinct target-bearing question/handle leaves** survived. The sole
lost target is `H200001` for question `gpt4_8279ba03`, atom
`appliance_smoker`. It was sealed as `definitely_irrelevant`; the only retained
leaf for that question was unresolved `H300000`, which describes a slow
cooker. The audit and byte-identical replay have artifact SHA-256
`a15e9b1058e0a86e51924f22aefa1b6858bcb08349c7a4a733ebe43b2cac23d2`.

This is a classifier/input-schema failure, not an A1a packing failure. The
question asks, “What kitchen appliance did I buy 10 days ago?” The exact
selected leaf already carries `date:2023-03-15` in its authenticated boundary
labels, but the classifier projection intentionally strips boundary labels and
supplies only H, G, leaf receipt, summary, and edge IDs. The true smoker leaf
and the competing slow-cooker leaf therefore could not be resolved by the
requested temporal offset from their provider-visible rows. The next revision
should derive a generic authenticated `evidence_date`/temporal-provenance field
from every leaf's boundary labels while leaving raw labels scheduling-only; it
should not tune terminal top-k or packing.

## Sealed inputs and outputs

| Artifact | SHA-256 |
| --- | --- |
| Sealed R7 construction and replay | `120199b9f4cf912b0a2c2d0b56b228813393a533da010909a92b4cd6268406a5` |
| A1 v2 construction and replay | `ad22a5b9c8d790f843de55c7653abdb9cbda9a7afb2661a67f3e50846bc37dca` |
| Accepted classifier dispositions and replay | `652b5f441f402d590e07bfb21130a436c8acb5666f0ac9b48bd657bce12ced5f` |
| A1a runtime construction and replay | `94be872da1f527927d20cf1b78de1d7c704ee6117e5976848a0b7239ee8e0ae2` |
| Treatment request population | `5a1b530fb7728620cd779f149e2a2f7bd9a36b240dfce0887e47d9ea26902625` |
| Fixed-union control request population | `adfb5db358faea31b64d94d736d0aa5b3a846e21abceb0cf37c65d5bbb29183d` |
| Post-seal target-retention audit and replay | `a15e9b1058e0a86e51924f22aefa1b6858bcb08349c7a4a733ebe43b2cac23d2` |

The runtime pair is sealed under:

```text
eval_results/matched_eval_100/locked-r7-a1a-raw-retained-terminal-preflight-v1/
├── r7-a1a-raw-retained-terminal-preflight-v1.json
├── r7-a1a-raw-retained-terminal-preflight-v1.json.sha256
├── r7-a1a-raw-retained-terminal-preflight-replay-v1.json
└── r7-a1a-raw-retained-terminal-preflight-replay-v1.json.sha256
```

The audit pair is sealed under:

```text
eval_results/matched_eval_100/locked-r7-a1a-postseal-target-retention-audit-v1/
├── r7-a1a-postseal-target-retention-audit-v1.json
├── r7-a1a-postseal-target-retention-audit-v1.json.sha256
├── r7-a1a-postseal-target-retention-audit-replay-v1.json
└── r7-a1a-postseal-target-retention-audit-replay-v1.json.sha256
```

## Runtime contract

For each question, A1a authenticates the A1 v2 construction/replay and the
classifier disposition construction/replay. It reconstructs the complete
`SelectedHLeaf` population and reruns the deterministic after-union semantic
partition. Every selected leaf must be accounted for as relevant,
definitely-irrelevant, or unresolved, with exact question, classifier-request,
leaf, and source receipts.

The treatment provider message contains the dated question, every R or U
summary in fixed-union order, its opaque H and G handles, and graph links whose
two endpoints remain. U is never silently top-ked or truncated. An empty or
over-budget retained union fails construction.

The fixed-union control uses the same system message, provider-input schema,
renderer, and source order. Its only experimental difference is that it does
not exclude I leaves. Each arm has independent presented-population,
provider-input, message, request, and request-population receipts. Control
requests remain marked
`sealed_control_non_actionable_until_paired_release`; this task did not execute
either arm. The pair permits a later causal 11-versus-11 answer assay without
confounding relevance filtering with a renderer change.

The hard contract is 8,000 total tokens including a 768-token answer reserve.
The largest treatment prompt is 1,588 tokens and the largest fixed-union
control prompt is 4,261 tokens. Both arms are under the cap for all 11
questions. Across the workload, the provider-input token proxy falls from
39,027 to 9,054, a reduction of 29,973 tokens; retained input density is
23.1993% by token proxy and 19.9475% by leaf count.

The runtime firewall excludes benchmark fields, ordinal routing, protected
parent predictions, references, source allowlists, semantic-atom manifests,
and target audits. It retains no provider or transformer state. Topic and
boundary metadata may affect upstream scheduling but does not have exclusion
authority.

## Per-question density and target retention

| Question ID | Union | R | U | I | Retained | Treatment tokens | Control tokens | Target atoms |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `d23cf73b` | 36 | 10 | 2 | 24 | 12 | 1,588 | 3,975 | 4/4 |
| `a9f6b44c` | 39 | 5 | 2 | 32 | 7 | 1,087 | 3,705 | 2/2 |
| `9d25d4e0` | 32 | 3 | 1 | 28 | 4 | 610 | 4,261 | 3/3 |
| `a89d7624` | 37 | 5 | 5 | 27 | 10 | 1,147 | 3,690 | 2/2 |
| `3a704032` | 36 | 3 | 2 | 31 | 5 | 599 | 3,804 | 3/3 |
| `gpt4_8279ba03` | 27 | 0 | 1 | 26 | 1 | 311 | 2,754 | **0/1** |
| `80ec1f4f` | 40 | 3 | 8 | 29 | 11 | 1,378 | 3,821 | 2/2 |
| `0a995998` | 34 | 5 | 1 | 28 | 6 | 921 | 3,607 | 3/3 |
| `1d4e3b97` | 39 | 2 | 7 | 30 | 9 | 872 | 3,368 | 2/2 |
| `9a707b81` | 34 | 2 | 0 | 32 | 2 | 430 | 3,550 | 2/2 |
| `7405e8b1` | 27 | 1 | 8 | 18 | 9 | 1,530 | 3,911 | 2/2 |
| **Total** | **381** | **39** | **37** | **305** | **76** |  |  | **25/26** |

## Evaluation isolation and promotion rule

The audit CLI authenticates a byte-identical runtime construction/replay pair
and its gold-blind firewall before opening the existing semantic-atom audit.
It then requires the audit and runtime to bind to the same sealed R7
construction. Target data cannot change the runtime leaf partition, prompts,
order, or receipts.

An atom is retained when at least one of its exact matching final H handles is
in the already-sealed retained partition. Promotion is strict: all 26 atoms
must survive, no target-bearing leaf may be pruned, and every treatment and
renderer-matched control prompt must be under the hard cap. The present result
fails the first two conditions and therefore remains NO-GO.

## Implementation and verification

- `tools/matched_eval/r7_a1a_raw_retained_answer.py` owns authenticated
  disposition replay, deterministic R+U selection, paired prompt construction,
  receipts, density metrics, cap enforcement, and runtime firewalls.
- `tools/run_r7_a1a_raw_retained_answer.py` seals and byte-replays the runtime
  artifacts without provider IO.
- `tools/matched_eval/a1a_postseal_target_retention.py` owns the isolated
  target-retention audit and GO/NO-GO rule.
- `tools/audit_r7_a1a_target_retention_postseal.py` seals and byte-replays the
  audit after the runtime seal exists.
- `tests/test_matched_eval_r7_a1a_raw_retained_answer.py` and
  `tests/test_a1a_postseal_target_retention.py` cover only-I exclusion, U
  preservation, no-top-k failure, paired rendering, cap enforcement,
  population/receipt binding, source/firewall isolation, target retention, and
  replay.

The exact provider-free production commands were:

```text
.pixi\envs\dev\python.exe tools\run_r7_a1a_raw_retained_answer.py --dispositions eval_results\matched_eval_100\locked-r7-after-union-a1-preflight-v2\terra-classifier-v1-network-recovery1\r7-after-union-a1-classifier-dispositions-v1.json --dispositions-replay eval_results\matched_eval_100\locked-r7-after-union-a1-preflight-v2\terra-classifier-v1-network-recovery1\r7-after-union-a1-classifier-dispositions-replay-v1.json

.pixi\envs\dev\python.exe tools\audit_r7_a1a_target_retention_postseal.py --runtime-construction eval_results\matched_eval_100\locked-r7-a1a-raw-retained-terminal-preflight-v1\r7-a1a-raw-retained-terminal-preflight-v1.json --runtime-replay eval_results\matched_eval_100\locked-r7-a1a-raw-retained-terminal-preflight-v1\r7-a1a-raw-retained-terminal-preflight-replay-v1.json
```

The focused runtime-plus-audit suite passed 17 tests with a unique base temp:

```text
.pixi\envs\dev\python.exe -m pytest \
  tests/test_matched_eval_r7_a1a_raw_retained_answer.py \
  tests/test_a1a_postseal_target_retention.py \
  --basetemp .test-tmp\a1a-boundary-20260830-h -q
```

The final compatibility slice passed 73 tests:

```text
.pixi\envs\dev\python.exe -m pytest \
  tests/test_matched_eval_after_union_fact_closure.py \
  tests/test_matched_eval_r7_after_union_a1.py \
  tests/test_matched_eval_semantic_binary_search.py \
  tests/test_matched_eval_typed_fact_compiler.py \
  tests/test_matched_eval_typed_operator_executor.py \
  tests/test_matched_eval_r7_a1a_raw_retained_answer.py \
  tests/test_a1a_postseal_target_retention.py \
  tests/test_matched_eval_artifacts.py \
  --basetemp .test-tmp\a1a-combined-20260830-a -q
```

## Next bounded experiment

Revise the selected-leaf/classifier provider schema to include authenticated
source or event dates, plus a deterministic relative-date projection derived
only from the dated question and leaf date. Reclassify the fixed 381-leaf
union under a new classifier format/root, preserve unresolved fail-open, and
repeat this same provider-free A1a construction and isolated audit. Do not
patch `H200001`, route on question ID/ordinal, or use the semantic-atom
manifest at runtime. Only after the treatment reaches 26/26 target retention
should the paired 11-versus-11 answer calls be released.

## Contamination caveat

The 26-atom result is post-seal and target-informed. It is valid only as an
evaluation of the already-sealed classifier partition; it is not a runtime
input and cannot authorize handle-specific repair. The diagnosis names the
lost target after the fact, but the proposed temporal-metadata repair is
generic and must be applied to every leaf/question under a new sealed
population. The current control prompts are sealed but unexecuted, so this log
does not claim an answer-quality gain or a benchmark score.

## Successor

Research Log 92 records the generic temporal fail-open composition. It applies
the same question-derived policy to every selected temporal leaf, retains
123/381 leaves, and passes the isolated post-seal gate at 26/26 atoms and 29/29
target-bearing leaves. Its promoted clean overlay/replay is `40a584d6…`, its
A1a runtime/replay is `e5d27693…`, and its audit/replay is `02d1a6f8…`. The
earlier `eb84f990…` envelope was withdrawn before compiler release after an
independent provenance review. This entry remains the immutable record of the
rejected 76-leaf arm.
