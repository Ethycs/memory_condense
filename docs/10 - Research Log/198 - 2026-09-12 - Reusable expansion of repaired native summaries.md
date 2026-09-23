# Reusable expansion of repaired native summaries

**Status:** R6 body admission and R4 exchange input preparation complete; expanded compilation queued. Another 34 rejected source batches repaired. The full native accuracy/latency target remains unverified.

The preceding goal turn made concrete progress through source repair and actual
partial-history routing verification. This continuation completed the new body
snapshot, admitted the completed recovered Qwen cache for reuse, and queued the
next complete expansion of the available bodies. None of these results is an
answer-accuracy or API-latency measurement. The target still requires all 100
separate memories to contain at least 1M eligible raw body tokens and to pass the
95/100 accuracy and eight matched latency gates in one fresh run.

## Completed admitted snapshot

`eval_results/native-spine-admitted-body-store-20260912-r6` completed normally,
with 13,468 bodies and 140,256 sections covering 139,638 original fragments.
All 11,438 R5 bodies and 119,144 prior sections are unchanged; 2,030 bodies were
added. All three bodies affected by malformed original batch 4541 are admitted.
Assembly used no model calls.

| Artifact | SHA-256 |
| --- | --- |
| `summary-bodies.json` | `37030e5fecf0ccbf1c807870f52f0abf4e61ad9c76700cfa3dc1adaad7680249` |
| `summary-bodies.sqlite` | `4b612aed3a2050a9c2b8c5d38d067061cd69b8df6941a6b97e18cbc93cc7c109` |
| `admission-snapshot.json` | `557458f0544cd8f0620865630b5d646806c9b157419e2faa93dbae235e9a845f` |
| `previous-snapshot-preservation.json` | `e904b8bf43bc8e9c262f2449cc34492e81b35546c738d758b3add33940b913da` |
| `handoff-finished.json` | `653ca132b992e964363741acf3ea507262f119421e45084f800445413bb0a84f` |

The snapshot is explicitly partial: 5,975 admitted batches, 7,816 pending and 21
unrepaired at its freeze. Session 45661 exited 0; PID 42240 has finished.

## Reusable exchange and hierarchy producers

`tools/compile_expanding_native_spine_exchanges.py` accepts completed original,
reused, recovered and expanding exchange producers. It replays their actual
request/response records, checks the same corpus and native backend, rejects
conflicts or incomplete sources, and reuses only identical neutral summary
requests. Its own successors can be reused by later expansions, so the same
producer supports the remaining source population. Accepted source records are
preserved; reuse does not manufacture new responses.

The real recovered R3 exchange source replays to the same result SHA
`614e928f05a13a3cbdc3881b4eca3162fb317f1905a435b37fe657dd97112758`.
All 317 accepted merge keys were admitted with the Qwen model unloaded and
generation explicitly forbidden. Verification SHA:
`bea081faf2bbc5546d18e4a8bab8cd45163529e51268b013a9f3eb43751295a6`.

The new inputs at `eval_results/native-spine-exchanges-20260912-r4` cover all
13,468 R6 bodies and 140,256 atoms. Preparation uses the JSON-aware body reader
before the existing exact-occurrence preparation path. It reads no benchmark
questions or gold and sends no raw text to Qwen.

- Inputs SHA: `266de080a6130b8da73270a543bcc4c2148a20fb9b3c15ea0171865bc3a1c067`
- Preparation policy SHA: `67250c9c8bb2c313d9d28df3e3502e505234fa82a21f233e97ee9bb026adc14b`
- Preparation finished SHA: `fd05e40a406da053acb20a0b3827eb225d2a4ee1495a64be294f8e4666b4aa86`
- Session 48999 exited 0; PID 69316 has finished.
- Exchange preflight SHA: `69f9162c6509042e798feb3750484634da1f7ef6e0703dd3a87d70d6e9471083`

The additional `--budget 0` pass completed 13,316 body exchange sets containing
69,529 exchanges, using the 317 accepted cache keys and no new local jobs or
batches. Another 152 bodies reached a pending merge request. Its partial result
SHA is `134ada4be4b58bbc0bc3449c975672e0192f1906c530a73ab6d3297532ee6430`.
Session 73828 exited 0; PID 61496, creation time `1789232245.362507`, finished.
The queued controller will replay this work before generating the remaining
summaries. This is complete processing of the zero-generation pass, not complete
exchange compilation of the 13,468-body snapshot.

The matching `prepare_expanding_native_spine_attention.py` retains the original
attention cache method and authenticates this new exchange producer.
`compile_expanding_native_spine_hierarchy.py` accepts completed recovered and
expanding parent ancestry as well as original parent caches. It retains the
128-token exchange and 512-token parent channel limits and exact raw addresses.
`ExpandingParentNativeSpineCorpus` authenticates the new parent producer and
JSON-aware body stores. Its actual serving verification remains pending until
the new attention and parent artifacts exist.

Twenty-four exchange/recovery tests pass, including all four reuse producer
types, unchanged old records, zero-call replay, rejection of corrupt or foreign
sources, incomplete and interrupted sources, and completion at a job-budget
boundary. Eighteen parent-reuse and namespace tests pass, including mixed
original/recovered/expanding ancestry and exact raw occurrence rebinding. The
new namespace constructor itself has not yet been exercised against a real
expanded parent result.

## Queued execution order

The live old parent compiler and vector R4 retain their current ownership.
The new stages wait for exact predecessor PIDs and creation times to disappear
and require matching terminal completion records before using the GPU. A timeout
does not restart a predecessor or discard its reserved work. Full-Qwen generation,
prefix attention and BGE encoding remain separate processes.

| Stage | Control root | PID / session | Policy SHA |
| --- | --- | --- | --- |
| Exchanges R4, at most 2,048 new jobs, 128 per invocation | `native-spine-exchanges-20260912-r4/continuations/2048-20260912-r1` | 49268 / 55723 | `efa24b02517e8f5a47262c3a748c2fee864c710295b38731eacce20b30c0a001` |
| Attention R5, existing cache method | `native-spine-attention-20260912-r5/handoff` | 42700 / 11051 | `4dfb9dbca67a4d253cf704d3b288e33bad63ff908d64126950f5c822c76fb51f` |
| Parents, at most 2,048 new jobs, 128 per invocation | `native-spine-expanding-parent-hierarchies-20260912-r1/continuations/2048-20260912-r1` | 54716 / 28744 | `8e5d947d9707511415ecbbe3ffd19ae35f57f69cd5a5e905f00d83d243e144a2` |
| Vectors R5 for R6 body summaries | `native-spine-summary-vectors-20260912-r5/handoff` | 17192 / 14889 | `323655bade9ca9e1b789de0b8ae116eb1fe6344107b97d3e0a830890d7ca4715` |

The exchange controller waits for completed input preparation and vector R4 to
exit. Attention waits for the new exchange controller to finish all prepared
bodies and exit. New parents wait for that attention stage; their reuse source
is the completed recovered parent producer, whose immutable preflight SHA is
`d6d6aa135c69575258cdc2c63935e0fef75cba078a58d071c9294b8d39873ba6`.
That source has now finished all 7,121 bodies in 540 new local jobs across this
continuation. Its result SHA is
`f51dfece2ca239ea70af36e3550bacce3a75b25d404cf9c1179b3c703303ae8b`; terminal
receipt SHA is `44612eab78012320346b4452bc63ec8694a5e8973556ebc35691a478db10cbb9`.
Session 64318 exited 0. The new parent reuse reader has also authenticated all
663 accepted parent merge keys with no model loaded and no new generation.
Verification SHA: `fc70ce10ef9bcc75db17daf330f69438b8230f3cf2a7ff5845c69639a1747313`.

Vector R5 first waits for vector R4 to finish, then prepares the R6 summary-vector
population while reusing the completed R4 vectors. Preparation keeps the encoder
unloaded. Encoding waits for the new parent process to exit, then uses BGE on
CUDA with batch size eight. Its preflight and vector count are not published yet.

Drivers are under `.tmp/`:

- `prepare_expanding_native_exchanges_20260912_r4.py`
- `run_expanding_native_exchanges_after_vectors_20260912_r4.py`
- `run_expanding_native_attention_after_exchanges_20260912_r5.py`
- `run_expanding_native_parents_after_attention_20260912_r1.py`
- `run_json_native_vectors_after_parents_20260912_r5.py`

## Further source repairs

The new selector classified 34 rejected originals outside the 252 previously
repaired batches. All were attributable section failures; no new malformed-JSON
or unclassified response was found. The first selector attempt stopped before
writing any artifact because a legacy result stores its source binding on each
admitted batch rather than on the aggregate result. Reading that actual binding
resolved the preparation error without changing any historical artifact.

R16 repaired 33/34 batches in 15 calls, retaining 718 valid original summaries.
One invalid new section remained. R17 replaced only that section with two exact
subdivisions in one further call. All 34 batches now pass: 764 original fragments
are covered by 834 final sections; 46 rejected fragments became 116 subdivisions.

- Selection SHA: `f9dbee5e3ce3342a01673d133e41ea4979b37da408f66a70fb8cc034f4a6d1cb`
- R16 preflight SHA: `04c337d588d1480291406c2dcc03e8771dc3781d37d1b02909110e11e325777a`
- R16 result SHA: `d5c265d83a1f96fdbfbb1cd84ee807ab2535aa72e372ec0924f5bce28352c6fe`
- Final root: `eval_results/native-spine-direct-section-repairs-20260912-r17`
- Final preflight SHA: `01c914e0e20d4be178e52980c2a0235f41d411a7b0a3c438cb38d66ad55bece2`
- Final result SHA: `c65c768f3a9a6d364556354ec3593e22406f6a410edf90d6b80936e0aadd6ffa`

These 16 calls used the authorized public raw fragments through the Terra
gateway. R17 is the sole final lineage for this cohort; do not also admit R16.
They are not included in the already frozen R6 store and need a later snapshot.

## Remaining work

Main ingestion remains live. Its latest counted original validations are 6,195
accepted and 293 invalid, 6,488 total; repaired original statuses remain unchanged.
The earlier 6,970-tree partial report remains the bound hierarchy input to the
readiness diagnostic below; the subsequent complete 7,121-tree result is
reported above.

Complete source ingestion and repair, final corpus admission, all hierarchy and
vector stages, and a fresh native full100 answer/latency run remain required.
The full100 loader now supports both recovered and expanding parent producers,
selecting their respective authenticating readers after the same early complete-
corpus gate. The 100-case population, actual minimum-1M token recount, fresh query
embedding, exact hydration, 400 fresh streams, 200 logical judgments, and joint
latency/accuracy policy are unchanged. The implementation binding now includes
the additional body-store and hierarchy producers. Forty-four focused native
population, full100 orchestration and streaming tests pass (72.58 seconds).

Before that loader edit, all 102 files bound by the original R1 readiness receipt
were verified and archived at
`eval_results/native-spine-joint-readiness-20260912-r1/implementation-before-expanding-loader.zip`.
Archive SHA: `454f781540a610ca2265d38a971ae386b0a3c4389230d7031f3001ff7fe688c0`.
The archive receipt SHA is
`d0ab4daf1d76acf76f23c1dbc50d13ce781e7d4a5e7d5663e0d8acca6b5ea1a8`.
That historical readiness result is unchanged and its exact implementation is
preserved. The live compilation controls do not depend on the evaluation loader;
their bound code remained unchanged.

A fresh real readiness check with R6 and the recorded 6,970-tree report still
rejects the incomplete corpus before any encoder, corpus adapter or provider
construction. It published no evaluation preflight and made zero model calls.
The diagnostic binds 113 implementation files and is not a benchmark pass:

- Root: `eval_results/native-spine-joint-readiness-20260912-r2`
- Readiness SHA: `820425819c0c59c9a50d97dd3b85c3be5907af3e58490080c38c67ebce1ee254`
- Actual admitted bodies: 13,468 of 31,166 required.
- Actual new-parent serving and the full fresh native answer/latency run remain
  unverified until their complete artifacts are available.
