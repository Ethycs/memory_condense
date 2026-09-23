# Native resident retrieval probe and direct source repairs

**Date**: 2026-09-12  
**Status**: Real resident component probe complete; repair and ingest work continue  
**Depends on**: [Research Log 189](189%20-%202026-09-12%20-%20Native%20namespace%20retrieval%20and%20reusable%20summary%20vectors.md)

## What changed

The prepared BGE stage finished all 17,146 unique summary embeddings. The native
resident path then performed 200 fresh query embeddings over 100 separate,
explicitly partial histories using the actual local model. Every accepted
baseline section was preserved by context expansion, and 5,584 hydrated spans
matched their original raw text exactly.

Warm retrieval took 0.105 seconds median for direct retrieval and 0.107 seconds
with additive chunk context. These measurements cover 34,462–90,984-token partial
histories, not complete 1M memories. There were no answer calls or judgments,
and no matched API timing control. This does not establish answer accuracy or
the joint target. The latest full100 answer result remains the earlier pooled
flat 84/100 versus hierarchy 8/100 failure.

The previous goal continuation made concrete progress by adding native serving,
checking real namespace isolation, and releasing vector execution. This turn
completed that local stage and its resident integration check, added direct
source-subdivision repair and admission, and started the next bounded jobs.

## Completed local stages

The previous resident Qwen continuation completed its entire allowance of 512
jobs in 128 batches. It ended at **209 body trees**, **348 leaves**, **139
parents**, and **824 original atoms**, leaving 1,460 first pending merge jobs
among the 1,669 prepared bodies. Source compilation is still incomplete.

| Artifact | SHA-256 |
| --- | --- |
| Final 209-tree partial report | `aef9df9994b4fc42c17cf48855f503102c6ca45eff8acc5740cd6f72f57020f2` |
| Qwen 512-job continuation completion | `67ec1066b716864fade4bca70aa9c968aa329c770ff21ee054e67a2dc8a6d1b3` |
| Native summary-vector result | `290231e50eba06edb80ceb8911bb9bed74ad7f30690183bd019b5c981152c90e` |
| Vector handoff completion | `142119e5275d692752d51834cf54def5e12024fca63084fdcd3e3c66ed37f3d2` |

The vector result is under `eval_results/native-spine-summary-vectors-20260912-r1`.
All 134 checkpoints contain normalized FP32 vectors with 1,024 dimensions.
They embed stored summary strings only and are reusable across actual source
occurrences. The handoff verified the previous Qwen process had exited before
loading BGE. Both previous jobs are terminal; sessions 56377 and 20807 must not
be restarted.

## Real resident probe

Root: `eval_results/native-spine-resident-serving-probe-20260912-r1`.

Preflight SHA:
`f80470c99775815222a63541283029b2d1248c21361733eea5977815c0c4aaf5`.

Result SHA:
`81565993f150b11d004ed259147a590f1e404d09c98f3975fb894f5021ab0755`.

Script: `.tmp/probe_native_resident_serving_20260912_r1.py`.

The script first replays the complete vector cache without loading a model.
It then explicitly warms the actual pinned BGE model and cold-loads each separate
namespace. Each arm performs one fresh query embedding. The paired order is
counterbalanced. Per-request intervals include live query encoding, summary
routing and raw hydration; cold model and namespace setup are excluded.

Four fixed generic component questions rotate across the histories. No benchmark
questions or gold answers are used. Both arms use the existing 3,072-token
rendered context limit and 128 raw-span inspection limit. Every partial history
still fails complete-1M admission.

| Component measurement | Direct | Additive chunk context |
| --- | ---: | ---: |
| Warm retrieval median, seconds | 0.104975 | 0.107289 |
| Warm retrieval p95, seconds | 0.166979 | 0.169157 |

All baseline evidence remains unchanged. Ten additional atomic sections fit in
seven histories. Limited spare context and the partial tree population restrict
additions; this is not evidence that they improve answers. Across both arms,
5,584 raw spans preserve their text and namespace ownership exactly. There are
200 live request embeddings plus one explicit warmup, zero query-time Qwen
passes, zero answer calls, and zero judgments. Session 53853 finished normally.

## Direct repair of rejected source fragments

The audited snapshot contains **79 unrepaired original batches** with **108
over-budget atoms** and **1,696 valid summaries** that must remain unchanged.
All failures are structurally attributable budget failures; there are no
malformed original responses in this selected population. The rejected raw
fragments total **64,622 token proxies**. The seven earlier repaired ordinals
are excluded from this wave.

`tools/repair_native_spine_sections.py` subdivides those rejected raw fragments
directly using the existing exact-offset subdivision routine. It avoids the
previous wording-only stage. It preserves all accepted original strings, packs
at most eight smaller sections per call, and checks normal response journals
before replay. It does not implicitly retry an unanswered request.

A successor repair root can refine only the still-invalid smaller pieces while
retaining accepted prior repairs exactly. The successor authenticates and
replays its predecessor before constructing the next bounded request population.
Each completed original batch is admitted independently; one remaining failure
does not hide other completed batches.

Root: `eval_results/native-spine-direct-section-repairs-20260912-r2`.

Preflight SHA:
`d6295592ecdab2d6af2d7e0c4f84ba8df1a30367705883d96443bb022b918758`.

Public source scope SHA:
`a3ab3be0c1f4d3d12200257d5321dea7d91122a6b69df4a4fafb3bb04f36fc1e`.

The exact prepared allowance is **35 Terra calls** for **280 smaller sections**,
through the previously authorized gateway. The scope binds the existing verified
public LongMemEval source provenance. Automatic approval review allowed execution.
No raw content is sent to Qwen. All 35 calls finished normally: **268 of 280
sections** passed validation and **69 of 79 original batches** became complete.
Result SHA: `9fdb2502ca4004d26221ccb14699f51051b335886de1fb9fe7909d795cdd383f`.
The result replays identically with 35 cache hits and zero new calls.
Session **31985** and PID **9884** are terminal and must not be restarted.

The follow-up at `eval_results/native-spine-direct-section-repairs-20260912-r3`
preserved all 268 accepted sections and subdivided the twelve remaining failures
into 29 pieces. Four new calls accepted 28 pieces, bringing the complete original
batch count to **78 of 79**. Session **54460** is terminal.

- Follow-up preflight: `68875800a4eda885370dcdcb0d341e33188425b562b0ea89572315388a9384db`.
- Follow-up result: `2ca145d3f6cd4ecc6ef36a75d7cc8249eebb06e409f572d9581145abcc0e815a`.
- Follow-up public scope: `203a3ce3c3b2c5d5a59a64720aab07bad61e98b881a09f881e590496fd1a48a9`.

The final `-r4` root subdivided the last 139-token-proxy section into three pieces
for one call, preserving all 296 accepted repairs. All three passed. **All 79
original batches are now repaired**, with 299 accepted replacement sections
and all 1,696 valid original summaries unchanged. The campaign used 40 new calls
in total (35 + 4 + 1), and the entire chain replays with zero new calls.
Session **75727** is terminal. The final preflight is
`38e4d200ae71ac908e3b643c59402b0ce32878c9e08086e6e32be7a68a936889`, with public scope
`5c1638ef0ac90be46482877431d5358d7b5a53d3b4cd3ebf3a79f95d67b611bf`.
Final result: `c4cf28af4d595ebd6bf9d281154bd4df7e405a6649ce260e416b253e15d95762`.
Use `eval_results/native-spine-direct-section-repairs-20260912-r4` as the complete
lineage in subsequent admission; do not also include its predecessors.

## Repair-aware admission successor

`tools/assemble_native_spine_direct_repairs.py` accepts both the earlier seven-batch
repair lineage and new direct-subdivision results. It replays each repair without
a provider, rechecks the original valid strings and exact coverage, rejects
duplicate lineages for the same batch, and excludes any body with a missing
fragment. Original source and validation files remain unchanged.

The body-store read format remains `native-spine-admitted-body-store-v1`, so
existing summary consumers can load it. This producer has a separate
`producer_format` and `producer_implementation` in both its snapshot and manifest;
it does not claim the old assembler produced the new database.
`DirectRepairSummaryBodies` checks this additional producer binding. The existing
base read-contract implementation binding is also retained and verified. Use
the successor assembler to replay these new repair lineages.

The first successor snapshot completed at
`eval_results/native-spine-admitted-body-store-20260912-r2` with explicit partial
admission. Its driver is `.tmp/assemble_native_after_direct_repairs_20260912_r2.py`.
It waited for the exact owned repair process to exit and for the bound result,
then assembled with zero provider calls. Pending or invalid bodies stayed excluded.

Handoff policy SHA:
`a813f22f6028f692156a27d4b0abc8614b85518ece0d12d2ca3a1e70d4311990`.

The handoff, session **48172** / PID **60648**, is terminal. Completion SHA:
`9c0a88817dbb55be05d5802707d4a709c4cb95f817e321b2a76d4cbf0d999a16`.
The body-store SHA is
`5f44877903788579374468efa6c28473fd262a4482e0972b2bfcf654fefc9441`.
It contains **5,243 complete bodies**, **54,753 sections**, and **54,590 original
fragments covered**. Seventy-six batches have repair overlays (the initial seven
plus the 69 completed first-wave batches); 28 original batches are still rejected
and 11,438 were pending at that frozen snapshot. This remains partial.

A separate snapshot at `eval_results/native-spine-admitted-body-store-20260912-r3`
has completed with the complete `-r4` repair lineage, plus the original seven
repairs. Its driver `.tmp/assemble_native_final_repairs_20260912_r3.py` binds the
completed final repair and the earlier store without relabeling either.
It performs zero provider calls and has already replayed the full repair chain.

- Policy: `7e717dab919dfd6d2ba1d99cb4074c80123f449978ccb5e759865805a704db8d`.
- Session **65758** and PID **4792** are terminal.
- Completion SHA: `88b51d7ab1ce9abb54e84cec74e7083d1f8dab06477b9e8b68fa8205a2f31d02`.
- Store SHA: `5c127b17048f22a0afed4d8b59b35e740ee6d0e945234433e195a9fba0365fcc`.

This snapshot contains **5,481 complete bodies**, **57,217 sections**, **57,015
original fragments covered**, and **202 additional exact subdivisions**. All
86 selected repaired batches are included. It admits 2,445 original batches or
repair overlays, with 22 original failures still unresolved and 11,345 original
requests pending at the frozen snapshot. Full source compilation remains false.

A direct comparison with the first admission snapshot verifies that all **1,669
previous bodies** and **17,190 previous atoms** remain byte-for-byte equal as
summary/pointer records. Preservation proof SHA:
`c5a8c2711025145567fd67d7b1b5aafbc8eaf91bd5383d0707975304eb4cd2b2`.

Do not restart either one-shot admission driver or reinterpret a frozen partial
snapshot as complete when the live source producer advances.

## Validation and remaining execution

Five direct-repair checks passed in 2.91 seconds (`77d38d`) and four admission
checks passed in 4.01 seconds (`7e609c`). They exercise normal checkpoint replay,
strict source preservation, progressive refinement, transport-failure handling,
unchanged valid summaries, partial/full admission, both repair lineages, read
compatibility, and database corruption. The cumulative native focused-check
count is now **136**. Synthetic clients are used in these tests; the real model
and source results are recorded separately above.

The next local Qwen continuation is running with a maximum of **2,048 additional
summary jobs**, still at most 128 per invocation with one resident model:

- Driver: `.tmp/continue_native_hierarchy_2048_20260912_r2.py`.
- Control root: the hierarchy root's `continuations/2048-20260912-r2` directory.
- Policy: `fad504659ad556a0a90dff73372c41fce3c84b2040834d51a1fdf6a4a29ce4ca`.
- Session **4888**, PID **67852**, creation time **1789213184.1045704**.

The exact process is live and producing accepted local merges. Do not load
another GPU model concurrently. A `stop-after-invocation` file requests a
bounded stop if the next vector stage needs the GPU.

That bounded stop has now been requested so the enlarged source snapshot can
be embedded next. The next vector preflight, at
`eval_results/native-spine-summary-vectors-20260912-r2`, is
`80d92193f8ec73af8e6ede4f447ed391ce39316efb5b5c58f16e41c296e68839`.
It contains **57,020 unique summaries**, reusing the earlier **17,146** vectors
and requiring **39,874** new embeddings. Preparation loaded no model.

Driver: `.tmp/run_native_vectors_after_qwen_20260912_r2.py`.
Handoff policy: `4a69bd0adcec630371969cf11b28682ab86daf3c499d7a2170f1250b317d071e`.
The exact live handoff is session **23061**, PID **59920**, creation time
**1789215082.4684079**. It waits for Qwen PID 67852 to exit with its matching
terminal receipt before loading BGE. Do not start another GPU process or rerun
this one-shot driver while it is pending or executing.

The main raw worker remains session **90673**, PID **65736**, creation time
**1789201959.128199**, verified live. At the recorded check it had **2,473 completed
batches**, **2,365 accepted**, **108 invalid**, and **55,414 accepted atoms**.
The original preparation contains 13,812 requests. Repair overlays do not alter
those original rejected-receipt counts.

Next: finish and replay this repair wave, refine its remaining failures, complete
the queued body admission, reuse existing vectors for the enlarged snapshot,
and continue complete-source compilation. The full100 answer and matched API
timing comparison still requires complete admitted 1M histories. Neither this
partial resident probe nor historical results may substitute for that gate.
