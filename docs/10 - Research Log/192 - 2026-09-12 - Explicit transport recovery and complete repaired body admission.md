# Explicit transport recovery and complete repaired body admission

**Date**: 2026-09-12  
**Status**: Six interrupted batches recovered; 40 more batches repaired; expanded snapshot admitted  
**Depends on**: [Research Log 191](191%20-%202026-09-12%20-%20Native%20benchmark%20support%20availability%20and%20ingestion%20continuation.md)

## Result and remaining scope

All six source batches interrupted by the original gateway failure now have
accepted replacement summaries. Their old request journals remain untouched.
Another 40 invalid-summary batches are fully repaired, preserving every one of
their 875 valid original summaries. The new body store admits **7,121 complete
bodies and 74,327 sections**, including all six recoveries and 126 repaired
batches across the accumulated repair campaigns.

This is still partial ingestion. Its fixed snapshot has **10,641 pending and
seven unrepaired batches**, out of the prepared 13,812. Main ingestion continues
independently. No new answer accuracy or matched API latency result was produced.
The 95%/1M/joint-latency goal remains unproven; the last native support diagnostic
was the explicitly partial 20-of-21 available-support result in Log 191.

The preceding goal turn made concrete progress through expanded vectors, the
real benchmark-question routing diagnostic and resuming unsent ingestion. This
turn likewise made progress: it recovered missing outputs, repaired rejected
source sections, added their authenticated admission path and built a larger
usable body store. Neither turn was blocked.

## Explicit recovery of retired requests

`tools/recover_native_spine_transport.py` accepts only explicitly selected
failed ordinals from a terminal compiler execution report. It reconstructs and
checks each old request's complete runtime identity and journal digest, verifies
that no original response or validation exists, and uses separate recovery
journals with their own provenance. Original journals remain terminal; an
ordinary restart still refuses to repeat them.

The original provider may have executed each unanswered request. The recovery
therefore records six uncertain original attempts separately from six new
responses; it does not describe the old attempts as zero calls or silently
replace their accounting. Each new recovery request uses the exact original
messages, the authorized Terra gateway, serial execution and zero automatic
retries. A failed recovery itself cannot be retried implicitly.

The driver `.tmp/recover_native_transport_20260912_r1.py` verified the original
executor had exited and bound the existing public-source proof before running.
Automatic approval review approved the six calls. Session 14474 is terminal.
Five responses passed immediately. Batch 2740 retained 21 valid summaries and
had one overlong summary for a 567-token raw fragment.

`tools/repair_native_recovery_section.py` replays the recovery output without a
provider, subdivides only its rejected fragment and reconciles exact raw
coverage while preserving all valid strings. One further call summarized three
smaller sections successfully. This yields **142 exact recovered sections over
16 raw bodies**, including two additional section boundaries. The original six
journals and the first invalid recovery response are all preserved.

| Artifact | SHA-256 |
| --- | --- |
| Recovery preflight | `eda8b8e133963bba7e4cb51c2e5ec803db82d94c0bbb4dff5c30e1b0463ee001` |
| Recovery execution policy | `801e1ca8884d49d01ed9a05ad6897ff022f8eed88b0f9176730c7a161eeb9942` |
| Six-response recovery result | `908e73d3ac0a87767d189000f93b7b3cf44fce31742012e22bf342bbe5d3244f` |
| Batch 2740 section-repair preflight | `2d5f31916bfb68f916aa48dc1f962f0191e6a4c0545bd971b105e5a10f611f9a` |
| Batch 2740 accepted section repair | `f394fef8f8e047b972214b75a79f2869546fedca42aa3dd0a9dd8cdc2f23a8ce` |
| Six-batch admission/raw-pointer verification | `b6f2cad5ee4d800ed7fafd7b963d6480b18668085809810d370a82b2f999bc67` |

Recovery artifacts are under `native-spine-transport-recovery-20260912-r1` and
`native-spine-recovery-section-repair-20260912-r1` in `eval_results`. The former
result truthfully retains `all_recovery_summaries_accepted=false`; the separate
section-repair result completes the final batch. Downstream admission combines
the five accepted recoveries and that explicit repair. Both stages replay with
zero new calls. Raw-pointer verification compared all 142 sections against the
unchanged source bank; it did not measure answer accuracy.

## Forty additional invalid-summary batches

The selected ordinals are:

```text
2052 2133 2143 2149 2158 2161 2186 2190 2209 2242
2251 2255 2288 2309 2324 2328 2340 2360 2370 2391
2392 2435 2493 2504 2520 2527 2529 2565 2595 2599
2626 2629 2714 2721 2728 2810 2869 2876 2889 2901
```

Their original outputs contain 875 valid and 50 invalid fragment summaries. No
structurally unattributable response was included. Existing direct subdivision
and refinement tools completed the campaign in **22 new calls**:

| Stage | New calls | Accepted replacement sections | Complete original batches | Result SHA-256 |
| --- | ---: | ---: | ---: | --- |
| Direct r5 | 17 | 119 | 35/40 | `b455b4ae7e7da5daa017da03f72f10f173cf2fa62adc349953d8b4e050b2fc5e` |
| Refinement r6 | 4 | 143 | 39/40 | `22320d8bfc565ca333ab6e217afd99cb061b8dc086f4041d99febe46de8d94ed` |
| Final refinement r7 | 1 | 147 | 40/40 | `aad3021778345fded2cbc70f9b20ccc061f870f3c5c7133415169c0292b930ec` |

The final 1,022 summaries cover 925 original fragments with 97 additional exact
subdivisions. All 875 valid originals and accepted earlier repair pieces remain
unchanged. Use only `native-spine-direct-section-repairs-20260912-r7` for this
campaign during admission; also supplying r5/r6 would duplicate its lineage.
Sessions 6342, 36253 and 41370 are terminal. All three stages replay without
provider calls. Their preflight SHAs are respectively:

- r5: `2cb0f96f87f42cfe565aebed420eb0c0ca236f1aaa597a0077406e9a3afb84b0`
- r6: `e142a413d02befbf0e4c811be0f5f171d1a3bf680fec8c2676a490f19fa3fe8c`
- r7: `4bb5965e1cf7af92fe82459a194303691b1cd60d4fbd381d6a54e81e601f1a97`

The recovery, its section repair and this campaign used **29 new calls in total**,
excluding independently continuing main ingestion. No raw content went to Qwen.

## Body-store and namespace admission

`tools/assemble_native_spine_recovered.py` adds a distinct producer supporting
normal outputs, prior summary-repair lineages, explicit recovery outputs and
repaired recovery sections. It independently replays every supplied lineage
with provider access disabled, rejects duplicate/cross-source replacements,
and excludes an entire body whenever any original fragment is unavailable.
It validates gap-free complete raw coverage before writing each body to SQLite.

The read-only wire format remains `native-spine-admitted-body-store-v1`, with
an additional producer binding `native-spine-recovered-body-assembly-v1`.
`RecoveredSummaryBodies` verifies that producer; the compatible base reader can
still read its unchanged wire format. `tools/recovered_native_spine_namespace.py`
provides an explicit corpus adapter for this producer, authenticating the old
template store and verifying preservation before binding expanded namespaces.

The new store is `eval_results/native-spine-admitted-body-store-20260912-r4`:

| Field | Value |
| --- | --- |
| Store SHA-256 | `7a049de472a5ea38e1efb8ebf0dd15b432ded29d1d364797afac2e955f835a67` |
| Snapshot SHA-256 | `1f8006e6232d265cf5e6d6a1aca1377e3629a40558daab93252ddd5f59c59709` |
| SQLite SHA-256 | `86ac234ffe4a9e539ca2d4f27a78310fc686cce7c1173ccd8e23f537f674bf16` |
| Complete bodies / sections | 7,121 / 74,327 |
| Original fragments covered / added subdivisions | 74,026 / 301 |
| Admitted / repaired / recovered batches | 3,164 / 126 / 6 |
| Pending / unrepaired batches | 10,641 / 7 |
| Full source / joint target passed | false / false |

Driver `.tmp/assemble_recovered_native_snapshot_20260912_r4.py` also checks that
all 5,481 earlier admitted bodies retain their exact summaries/pointers, then
prepares a vector successor reusing the completed 57,020-summary vector store.
Vector preparation does not load a model or claim embeddings are complete.
Its handoff policy SHA is
`56c88872c600c7fd6b83bc5bd397447fc35e526f13964b6047394ae15c7dd635`.
Session 93657 exited successfully. Its handoff completion SHA is
`50899cbb60631f7bfb76b4df0ba72a5a4cd495033f65c53177993a5ad7c02a2a`.
All 5,481 prior bodies / 57,217 sections were verified unchanged; the snapshot
adds 1,640 complete bodies. Preservation receipt SHA:
`f3118abd22ffd1804c2fec09c5d7c5be5fabc077e58a78e819271c373fa55b92`.
The next vector preflight, under `native-spine-summary-vectors-20260912-r3`, is
`0688f204ad21def661e2ec0e59f7b676cd9afb2aca76f7bb2d867e2224c79316`.
It prepares 74,059 unique summaries, reusing 57,020 existing vectors and requiring
17,039 new embeddings. This preparation left the encoder unloaded; vector
execution has not been released while the current Qwen job owns the GPU.

`.tmp/verify_recovered_native_serving_20260912_r1.py` completed successfully,
checking all 142 recovered atoms through 15 actual namespaces and the existing
exact hydrator, using single-atom summary self-queries. Every checked namespace
was partial and rejected full1M admission. This is a component check with no
benchmark questions, model calls, judgments or accuracy/latency claim. Its root is
`native-spine-recovered-serving-verification-20260912-r1`; preflight SHA
`d13d0db058eeae2d6b83dc51b0b6c24a9254ea230f12f6c5e3749ec4037d1ce9`, result SHA
`5486355b12972afc657d15a182afda69997e1b4dc53f7fdd12445d7ade742570`.
Session 91104 is terminal, and all 24 bound implementation files were verified
unchanged after completion.

The expanded exchange-input preparation also completed, without model calls:
7,121 per-body inputs contain all 74,327 admitted atoms at authenticated actual
source occurrences. Root: `native-spine-exchanges-20260912-r2`; inputs SHA
`49e5665e357baf27e698eba4fc00799ef13c8342ca4b4031c85b5bcaf073fcc9`;
completion SHA `34c545c81b3c0f2dc54d16beb586bdf6d08560cb1513d1100faadd75489c0317`.
Session 96540 / PID 69896 is terminal. Driver
`.tmp/prepare_expanded_native_exchanges_20260912_r2.py` bound the recovered store's
producer independently before invoking the unchanged input preparer. These are
prepared inputs, not completed exchange summaries or hierarchies.

## Checks and ongoing jobs

Seven transport-recovery tests pass, including unchanged original journals,
separate accounting, zero-call replay, corrupt-journal rejection and terminal
replacement failure. Six recovered-admission tests pass, including combined
repair/recovery coverage, partial/full admission boundaries, invalid output,
duplicate lineage, corrupt database and subdivision of a recovered section.
This adds 13 distinct checks, bringing the native cumulative count to **160**.
The four original admission checks were rerun after adding section-repair support;
that rerun is not counted as four additional tests.

The frozen recovery (eight files), recovery-section repair (11 files) and direct
repair (12 files) implementations were checked unchanged after their real runs.
Original raw, hierarchy, vector and previous evaluation implementations remain
untouched. Existing staged work is preserved.

Main ingestion session 98453 / PID 56400 / creation time 1789216671.9337437 and
Qwen session 95218 / PID 67944 / creation time 1789216584.6193802 were both
verified live. Main original validations reached 3,051 accepted and 133 invalid
(repair overlays do not relabel these). The latest inspected Qwen report has
314 trees, 740 leaves, 426 parents and 1,744 original atoms, SHA
`7f6319049f5dea16b9ef8c3a9b057066b242b261cad8b8660d7d5eaf448f4a9d`.
It still compiles the earlier 1,669-body hierarchy population. Source ingestion,
subsequent hierarchy expansion and the fresh full1M answer/latency evaluation
remain necessary. Do not load another GPU model while the exact Qwen process
owns that stage.

## Next bounded hierarchy assessment

Code inspection found that `build_user_spine_hierarchy` uses one
`max_channel_tokens` value both to bound generated parent channels and to check
the attention input cap. Its attention calls actually read the original compiled
exchange user summaries before building any parents. This suggests a testable
way to reduce ingest generation: keep the original 128-token exchange summaries
and exact cached attention inputs, while allowing larger bounded parent channels
to reuse more existing text.

A successor should first measure whether separating those budgets reduces
pending Qwen merges on the same real exchange population, while proving identical
attention receipts, cuts and raw-span coverage. This has not been implemented or
measured; it is not a speed or accuracy claim. Do not change the frozen builder or
current preflights, silently enlarge the scorer's advertised capacity, or feed
longer input text through a cache keyed for the old attention windows. Completed
native summaries and accepted neutral merge results should be reused explicitly
when expanding compilation to the new 7,121-body inputs.
