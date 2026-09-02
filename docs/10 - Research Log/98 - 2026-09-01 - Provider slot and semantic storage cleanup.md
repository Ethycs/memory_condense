# Provider slot and semantic storage cleanup

**Date:** 2026-09-01

**Status:** implemented, provider-free, and regression-tested; no retrieval or
answer-score claim

## Question

Are the current memory and terminal paths leaking protected data, retaining
query data too long, or spending provider and process memory on duplicated
identity and tree structures?

## Result

No gold, reference answer, raw source/chunk ID, filesystem path, or retained
model state was found in the active provider messages inspected. There were,
however, three concrete cleanup targets:

1. specialist prompts exposed stable candidate/group/slot hashes and repeated
   inventories beside their opaque provider handles;
2. transition-trace evaluation retained raw texts in large process-global LRU
   caches, while the residual classifier retained node conversions that had no
   reuse; and
3. every semantic-tree node retained a tuple slice of all descendant cells.

All three are now repaired under compatibility constraints.

## Provider result

Historical `COMPACT_FINAL` remains byte-identical compact v1. New
`COMPACT_FINAL_V2` uses short S/K aliases, derives represented handles from the
binding plane, moves repeated defaults to one header, and keeps stable
candidate and raw slot identities local. Specialist advisories expose H handles
only. The local validator remains dual-schema and reconstructs synthetic local
proof IDs after the provider boundary.

On the same ten sealed specialist inputs:

| Measurement | V1 | V2 | Saving |
| --- | ---: | ---: | ---: |
| Provider JSON tokens | 37,643 | 25,687 | 31.76% |
| Full chat tokens | 40,783 | 28,827 | 29.32% |
| Provider bytes | 140,924 | 104,876 | 25.58% |

No provider call was made. The exact token saving is 11,956, or 337--1,799 per
question.

## Memory result

Transition scoring now owns a call-scoped cache and releases it after the arm
score. Residual classification keeps the useful one-per-query manifest map but
does not retain per-node term/action caches. Sealed records no longer inherit a
`__dict__`.

Semantic nodes share one immutable population and use spans internally while
preserving their historical public `.cells` view and exact projected identity.
At 1,024 cells, descendant tuple storage falls from 171,992 to 8,232 bytes
(-95.2%); combined node/population/preorder storage falls from 368,544 to
221,160 bytes (-40.0%).

## Verification

- 101/101 combined integration tests passed.
- 878 passed and one skipped across the complete matched-eval suite.
- 83/83 downstream construction, replay, full-100 sidecar, specialist-final,
  answer-loader, and historical prompt compatibility tests passed.
- All changes are provider-free; no journals or sealed historical artifacts
  were rewritten.

The complete matched-eval run took 443.90 seconds. A duration assay attributes
235.11 seconds to the frozen q82 V6.1 terminal contract and 39.08 seconds to one
exact-span dedup fixture. This is consistent with the prior SQLite-heavy fixture
diagnosis and is not evidence that semantic query latency regressed.

## Remaining work

Visit records still retain span-wide coverage IDs, and node manifests retain
Python-float centroids. Both require explicit successor containers to preserve
`dataclasses.replace`, pickle, equality, certified bounds, and legacy receipt
semantics. The locked full-100 terminal also remains compact v1; a new named
construction must opt into v2 rather than changing that sealed lineage in
place.

The next meaningful experiment is therefore a compact-v2 semantic-global
successor that uses the recovered slots for additional evidence under the same
8k cap. Smaller prompts alone are an apparatus result, not improved recall.
