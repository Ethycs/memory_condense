# Native hierarchy compilation and atomic evidence recovery

**Date**: 2026-09-12  
**Status**: Component verification complete; bounded local compilation continues  
**Depends on**: [Research Log 187](187%20-%202026-09-12%20-%20Native%20user-spine%20exchanges%20and%20reusable%20Qwen%20attention.md)

## Result and limits

The native compiler now builds source-local hierarchies from the completed
user-spine exchanges and cached Qwen attention. Each completed body tree retains
the entire original atomic index for fine retrieval and evidence-budget recovery.
The first bounded run completed 167 trees. A subsequent resident continuation
has reached 196 complete trees; it is still running at this checkpoint.

This is ingest and component work. The corpus is incomplete, no fresh native
full100 answer score exists, and neither the 95% accuracy target nor the matched
API latency target has passed. The prior pooled full100 remains flat 84/100
versus hierarchical traversal 8/100, as reported in Log 181.

## Implementation

`tools/compile_native_spine_hierarchy.py` authenticates the completed exchange
result by zero-call replay and reads only the already-computed attention windows.
`FrozenAttention` rejects unknown windows, changed cache artifacts, changed
checkpoint identity, and changed precision or attention settings. It has no
encoder and cannot initiate an additional attention pass.

Existing user-spine hierarchy construction uses those signals for binary cuts,
with a 512-token raw leaf target, at most two exchanges per leaf, eight-exchange
attention windows, and 128-token summary channels. Oversized exchanges remain
indivisible and are diagnosed. Date-neutral summary merges use the direct local
Qwen backend. Qwen receives summaries only. Original raw spans and source dates
remain in the authenticated descriptor layer.

Completed `hierarchies/<body_sha256>.json` files include both `index_json` and
`atomic_index_json`. A tree is published only when every required merge exists
and its single root partitions all original spans in order. Partial progress
reports remain explicitly partial.

`src/memory_condense/search/native_hierarchy_occurrence.py` rebinds a completed
body tree to another actual source occurrence without model calls or raw reads.
It preserves summary channel text and topology, verifies every original atomic
pointer against the new occurrence, and renders the actual occurrence date.

`src/memory_condense/application/atomic_section_fallback.py` recovers pre-ranked
original atoms belonging to budget-rejected coarse sections. It retains all
accepted baseline evidence, shares the raw-inspection budget and turn cache,
preserves the original coarse rejection diagnostics, and does not bypass missing
or stale raw-source failures. It never clips raw text or claims factual closure.

## Real artifacts

Root: `eval_results/native-spine-hierarchies-20260912-r1`.

| Artifact | SHA-256 |
| --- | --- |
| Hierarchy preflight | `43fd03942328313f6264e490ebb65799908741965ebac9899abcad3aa4e0add2` |
| First 167-tree partial report | `0c8b4cddf2c4d222c33987a6276a183d85d8829b2fd2047789f4c021781a1c1f` |
| 170-tree partial report | `2f8c31471dc95860d42268a508140351cb275fca26ee8f8953b76c7020e96085` |
| 181-tree partial report | `89f4e1435c41439bad6c87c1f9798f456bace8fee57e4d1ce3dc0551e37def72` |
| 196-tree partial report | `5f6494f82bdae0ce7513edcf91bc7c5c16268d2e200280b544f03a70350db32f` |
| Serving component verification | `ee9a531907a19162b513a6c8761e7ff5ab83788d9ff03791c32a0b9a243f887f` |

The first 128 new local merge jobs used 32 batches. Of those jobs, 107 produced
accepted new cache values; sixteen accepted exchange-stage merges were reused.
That report contains 167 bodies, 225 leaves, 58 parents, and 516 original atoms.
Later jobs fill intermediate dependencies before additional trees become whole;
job counts and complete-tree counts therefore differ substantially.

The real serving component check rebinds all 167 initial templates at 314 actual
source occurrences. Eighty-six bodies appear at multiple actual dates. All 940
tested raw spans hydrate exactly. All six exchanges exceeding the 3,072-token
reader budget recover their preselected final original atom through the fallback
within that budget. Those six probes intentionally select the atom and use its
summary as the query: they measure budget recovery, not retrieval accuracy.

Script: `.tmp/verify_native_hierarchy_serving_20260912_r1.py`.

## Focused checks

Fifteen checks pass: three in `test_native_spine_hierarchy.py`, six in
`test_atomic_section_fallback.py`, and six in
`test_native_hierarchy_occurrence.py`. They cover bounded compilation and replay,
cached-attention integrity, exact occurrence rebinding, budget sharing,
preservation of baseline evidence, stale-source rejection, and cross-occurrence
isolation. The synthetic attention fixtures do not measure GPU behavior.

This brought the cumulative native focused-check count from 96 to 111.

## Resident continuation

Driver: `.tmp/continue_native_hierarchy_512_20260912_r1.py`.

Control root:
`eval_results/native-spine-hierarchies-20260912-r1/continuations/512-20260912-r1`.

Policy SHA:
`53944a82874cd643a414f8d5e963b0766c16df6223cd4545594d70162b3d1721`.

The driver allows at most 512 additional local merge jobs in invocations of at
most 128, keeping one Qwen model resident. It is session **56377**, PID **63632**,
creation time **1789210114.340639**. The PID and creation time were verified live.
There was no terminal `finished.json` or `failure.json` at the latest check.

The 196-tree report contains 307 leaves, 111 parents, and 723 original atoms.
It follows three completed 128-job invocations in the resident continuation.
Do not restart the one-shot driver or load another model on its GPU while its
exact process remains live. A `stop-after-invocation` file requests a bounded
stop after the current invocation, if needed. Log 189 records the prepared BGE
handoff that waits for this process to finish before loading its model.
