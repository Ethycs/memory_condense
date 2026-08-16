# Source heat diffusion: dual QK exploitation plus bounded source exposure

**Date**: 2026-08-16
**Status**: DEVELOPMENT / POSTHOC REPLAY — implemented and locally measured;
not a fresh confirmation or a public-benchmark result
**Predecessor**: `04 - 2026-08-16 - Safe associative memory confirmation.md`

## Question

Can the compiled attention graph behave like a live memory allocator: diffuse a
query's heat through compact QK/OV associations, select the next useful item,
and make the amount of text shown from each source proportional to its heat—
without loading Qwen during reads or retaining transformer token state?

## What was built

`heat_diffusion.py` implements a finite, personalized-PageRank-style walk over
the persisted association graph. Hybrid anchors seed one unit of scalar heat.
At each hop:

1. a fixed restart fraction returns to the query anchors;
2. the remaining heat follows row-normalized, temperature-scaled stored-edge
   utility;
3. heat from multiple parents is summed;
4. only a fixed number of chunk IDs, scalar values, and one explanatory path
   survive to the next hop; and
5. retained heat is renormalized after the frontier cap.

There is no Qwen call on this path. No token K/V, hidden-state sequence, or
candidate text is carried by the diffusion loop. Text is hydrated from the
external chunk store only after a bounded set of chunk IDs has been selected.

The first implementation also connects heat to prompt construction. Candidate
heat is aggregated by memory source (currently `turn_id`), candidates are
ordered by heat per token with weighted-fair source scheduling, and
`ContextPacker` can enforce the same source-aware expansion order under its
hard token budget. The low-level API accepts a different source-key resolver,
so a document, conversation, user, or hyperedge can replace `turn_id` later.

## Why the selected arm has two channels

Pure diffusion rewards corroborated, token-efficient paths. It can nevertheless
dilute a rare but decisive max-path edge. The selected policy therefore spends
two association slots as:

- **one ranked-QK exploitation slot** — preserve the strongest learned local
  association; and
- **one heat slot** — explore the diffused graph and balance source exposure.

This is not an ensemble of models. Both channels read the same compact external
association artifact and ordinary reads remain model-free.

The selected operating point is two hops, three neighbors per node, a frontier
cap of 16 IDs/scalars, restart probability `0.20`, edge temperature `2.0`, a
`0.90` lexical-anchor guard, no prompt-token increase, a `0.60` per-source
expansion cap, and a 1,024-token packing budget. Degree-two pruning is the
selected storage variant.

## Results

All numbers below are replays over already-consumed local splits. Recall is
answer containment at `k=5`; token counts are the selected chunk-text proxy
reported by the existing rig. They are useful for choosing a development
policy, not for estimating public performance.

### Selected two-hop dual channel, before physical pruning

| Local split | Questions | Hybrid recall | Dual recall | Hybrid mean tokens | Dual mean tokens | Reduction | Recovered / lost |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Development | 12 | 83.3% | **91.7%** | 989.8 | 833.4 | **15.8%** | 1 / 0 |
| Replay v2 | 18 | 100.0% | **100.0%** | 1,032.4 | 923.0 | **10.6%** | 0 / 0 |
| Locked-confirmation replay v3 | 18 | 83.3% | **83.3%** | 973.9 | 927.6 | **4.8%** | 0 / 0 |

The one recovery is the same development recovery already found by ranked QK;
heat did not create an additional recall gain. It reduced the text cost while
preserving that exploit path.

### Selected two-hop dual channel, degree-two graph

| Local split | Linked recall | Mean tokens | Reduction vs hybrid | Directed edges | Token-state bytes |
| --- | ---: | ---: | ---: | ---: | ---: |
| Development | **91.7%** | 828.7 | **16.3%** | 738 → 492 | 0 |
| Replay v2 | **100.0%** | 927.0 | **10.2%** | 1,420 → 952 | 0 |
| Locked-confirmation replay v3 | **83.3%** | 924.1 | **5.1%** | 1,204 → 812 | 0 |

Physical pruning preserved recall on all three replays. It slightly improved
tokens on development and v3 and slightly worsened them on v2, so the claim is
storage reduction with observed recall non-regression—not universal token
improvement.

## Negative result that changed the design

On development, pure three-hop heat used 798.6 mean tokens (19.3% below the
hybrid baseline) but stayed at 83.3% recall and failed to preserve ranked QK's
one recovery. A three-hop dual arm likewise fell back to 83.3%. The additional
walk depth was therefore rejected. The evidence favors a shallow graph with an
explicit ranked exploitation reserve, not increasingly recursive traversal.

## Decision

Keep the two-hop dual channel as the development default for the next fresh
evaluation. Keep the frontier at 16 because it contains only external IDs and
scalars, not 16 hydrated memories or transformer activations. Keep degree-two
physical pruning enabled in that next comparison.

Do **not** claim a new recall improvement. The next valid gate is a new locked
source-family split or a public common benchmark with equal token budgets.

## Reproduction record

- Artifact ID: `assoc-5d0da26f03437cd9be7df2a1`
- Dual-channel arm hash:
  `1fd2ef13fc36587c00dcab00de87561d9bd05dd7b1a466fddcadf1ef1fa0f7cb`
- Degree-two follow-up arm hash:
  `a366f20a8e263d3ed413c92d78e5088a4e973becead7224b41ef23263e5d13ba`
- Configs:
  `tools/performance_rig/configs/heat-diffusion-dual-channel-arms.json` and
  `tools/performance_rig/configs/heat-diffusion-dual-prune-arms.json`
- Reports:
  `C:\Users\Keytone\Downloads\memory-condense-rig\sweeps\20260816-heat-dual-*`
  and
  `C:\Users\Keytone\Downloads\memory-condense-rig\sweeps\20260816-heat-dual-prune-*`
- Machine-readable summary:
  `eval_results/qwen_source_heat_diffusion_development_summary.json`

Verification completed with the normal Pixi environment:

```powershell
pixi run --frozen -e dev pytest -q
```

Result: **627 passed**, one pre-existing Pydantic settings warning, in 108.15 s.
