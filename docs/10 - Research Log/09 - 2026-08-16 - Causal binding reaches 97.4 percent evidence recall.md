# Causal binding reaches 97.4% evidence recall

**Status**: development literal-recall result; not answer-stage judged accuracy

## Question

Can a live, external consolidation graph use transient Qwen QK/OV activity to
improve the operational outcome test—recovering the evidence needed after a
long chat without sending the full transcript—while keeping transformer memory
bounded and preserving direct-retrieval evidence?

## Locked inputs

- corpus: `data/build-session-8f7f7561.store`, 2,420 chronological turns;
- corpus database SHA-256:
  `58b12e1a978304a799050d16488477c5e66fe8549fb4efb9de2ce308b80f0278`;
- probe: `data/2026-08-16-build-session-baseline/cc_probe.json`, 39
  verified literal questions;
- probe SHA-256:
  `2669b73c9153325727724606a6bcdebf2c4ec10cbb8fecec1776d9f0ca8080fe`;
- all arms use hybrid retrieval `k=10` and the same 1,600-token evidence cap.

The output is isolated at:

```text
C:\Users\Keytone\Downloads\memory-condense-rig\runs\consolidation-replay-20260817-003652
```

## What changed

The first two chronological replays exposed two write-policy defects. Selecting
only retrieved old chunks never admitted newly experienced answers. Selecting
the five response chunks closest to the initiating prompt still dropped sparse
tool outcomes. Requiring two observations for every edge then made unique
episodic facts unreadable even when they were admitted.

The corrected path is:

1. retrieve prior context before ingesting the current user prompt;
2. store the prompt and all following assistant/tool/system chunks;
3. cover every outcome through fixed slices of at most nine nodes, including
   the initiating prompt and prior anchors;
4. let Qwen inspect only three candidates per disposable pass;
5. retain typed IDs, decayed scalar masses, ordinary co-access counts, and a
   schema-v9 `causal_count`—never tokens, activations, residuals, or KV state;
6. admit a unique completed prompt/response binding after one causal
   observation while ordinary old-to-old co-access still needs two;
7. add learned candidates instead of evicting the last direct result;
8. run two bounded scalar diffusion hops, rerank durable chunk IDs against the
   live query, and balance three read slots across hop depths;
9. pack by retrieval utility divided by square-root token cost under the same
   hard evidence-token ceiling.

The rank and Qwen graphs received identical event memberships. An event was
applied to the rank arm only after the matching Qwen inspection succeeded, so
failures could not give one arm more training data.

## Result

| Arm | Literal hits | Recall | Mean evidence tokens | Graph-read probes |
| --- | ---: | ---: | ---: | ---: |
| original operational pack | 35/39 | 89.74% | 1,418.5 | 0 |
| budget-aware packing only | 36/39 | 92.31% | 1,349.0 | 0 |
| rank causal consolidation | 37/39 | 94.87% | 1,431.9 | 35 |
| Qwen causal consolidation | **38/39** | **97.44%** | **1,423.8** | 36 |

There were no losses relative to the original arm.

- Packing alone recovered q27 and saved 69.5 mean tokens.
- Rank consolidation additionally recovered q0.
- Qwen additionally recovered q38, used 8.1 fewer mean tokens than the rank
  graph, and was the only arm above the 95% literal-recall threshold.
- Qwen cost 5.3 more mean evidence tokens than the original arm, or about 0.4%,
  while remaining below the unchanged 1,600-token cap.

The Qwen-specific q38 result is consistent with the intended iterative linker:
the answer was not a one-hop candidate, but was reachable in two scalar graph
hops. Hop-balanced query reranking selected it in the Qwen-weighted graph; the
rank graph did not.

## Runtime and bounds

- 41 completed eligible episodes;
- 2,320 outcome chunks covered by 483 bounded events;
- zero failed Qwen events;
- Qwen update time: 324.35 seconds total, 0.672 seconds/event mean;
- frozen query embedding: 14.37 seconds;
- chronological staging: 35.80 seconds;
- Qwen prefix load: 2.77 seconds;
- Qwen graph: 2,409 nodes, 7,081 edges, 483 bounded receipts;
- retained prompt/transformer state: **0 bytes**.

The teacher is the official Qwen checkpoint loaded as a seven-layer BF16
prefix. This run did not load the full 8B model, did not use an FP32 checkpoint,
and did not call central-dev or any paid model endpoint.

## Remaining miss

q13 asks which text filled the 60 smoke-test turns: `unrelated chatter`. The
answer chunk is now present in the graph, but it is in a component not reachable
from q13's direct seeds even within five hops. More read depth cannot repair
that. The next consolidation improvement should preserve connectivity across
large tool episodes—for example a bounded rolling episode bridge or a compact
episode node—without recreating transcript context or relaxing degree bounds.

## Reproduction

```powershell
pixi run -e dev qwen-consolidation-replay `
  --source-store data/build-session-8f7f7561.store `
  --probe 'docs/10 - Research Log/data/2026-08-16-build-session-baseline/cc_probe.json' `
  --output-root 'C:\Users\Keytone\Downloads\memory-condense-rig\runs' `
  --embedding-device cuda --qwen-device cuda `
  --expansion-tokens 1600 --retrieval-k 10 `
  --max-event-nodes 9 --new-event-nodes 5 `
  --qwen-group-candidates 3 --max-prompt-tokens 128 `
  --max-workspace-tokens 1024 `
  --consolidation-read-slots 3 --consolidation-hops 2 `
  --consolidation-candidates 128 --consolidation-diffusion-width 32 `
  --budget-aware-packing
```

Verification after implementation: `742 passed`, with one unrelated
third-party Pydantic settings warning.

## Interpretation

The 95% development evidence-retrieval target is met on this locked chat, but
the primary target remains answer-stage judged LongMemEval accuracy. The next
assessment should freeze this policy and test whether the answer model can
produce the correct outcome from the packed evidence, without the transcript.
