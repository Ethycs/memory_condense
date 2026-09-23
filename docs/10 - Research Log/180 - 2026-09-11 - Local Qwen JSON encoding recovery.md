# Local Qwen JSON encoding recovery

**Date:** 2026-09-11  
**Status:** Successor compilation and full100 handoff running; joint target unproven

The four-summary compiler stopped on a JSON encoding error in the fourth memory.
A successor removes only invalid apostrophe escapes inside JSON strings, checks
the result with the unchanged summary contract, and preserves the original
response and a reproducible repair receipt. It recovers the failed saved summary
without another model call and continues the same Qwen generation configuration.

## Failure and bounded repair

Original compiler session 65191 exited 1 after 2,315 generation jobs in offset
030. Its final checkpoint had 345 complete source trees and 1,483 parents.
The final failed response used `\'` inside a double-quoted JSON string in the
song title “Sweet Child O' Mine.” Removing the backslash at character position
87 lets the existing strict parser accept a 49-token summary under its original
128-token limit. No summary wording was rewritten.

The saved request SHA is
`a28b92f3cf78be3a3af562eb68021ebe19092d5e1487a9f0a36d2551932abcd8`;
response SHA is
`3c168f52dcd8f4dd11e60f46f590bbff25f3a591d3d21fccf92d185bbccd7d24`;
job SHA is
`c51738f50d47be6c95188621309e11f3819795247acbef68ffa9ca19ed9719fb`.
Original handoff session 61096 also exited 1 after rejecting the incomplete
population, before readiness or evaluation calls. Both original processes were
confirmed terminal before successor preparation.

`tools/local_spine_json_recovery.py` implements a JSON-string scanner. Valid
JSON and legitimate escaped backslashes remain unchanged. The repair refuses
other invalid escapes, truncated JSON, duplicate keys, extra fields, oversized
summaries and generation without EOS. Future repaired rows retain the exact
original response, original generation token counts, changed positions and
before/after text hashes. Semantic recovery remains bounded to two attempts;
encoding repair makes no model calls.

## Preserved work and preparation

`tools/run_local_spine_json_recovery.py` requires the original compiler and
handoff to be terminal, then authenticates and snapshots their completed work.
It imported 7,627 additional valid local summaries, with zero unacknowledged
requests. One further summary was recovered by encoding repair. Including the
inherited caches, the new ten-memory snapshot contains 8,848 summaries.

Preparation ran with generation forbidden and a zero-job allowance. Offsets
000, 010 and 020 reconstructed completely with identical summary text, source
and section IDs, child relationships and raw spans. Every original leaf remains
identical. Parent summarizer identities now bind the successor preflight, so
the hierarchy artifact hashes correctly change. The model was never loaded.
Preparation session 32015 exited 0 (chunk `be7d26`).

The fourth memory resumes from 413 complete source trees and 1,890 parents,
with 60 next dependencies. Its preflight SHA is
`d3088fc4f35adab8034bd73f57d8a7abfa04b917dc7c17543025e9cc5560e327`,
and its prepared progress SHA is
`399c7303a56aa9fc0bd63d371400a11930542e5a91043bd8ae309a20a9eaf396`.
An independent replay of the changed input cache produced the identical cache
with zero model calls (chunk `1db00d`). This verifies encoding and reconstruction;
it does not establish semantic fidelity or answer accuracy.

Artifacts are under `eval_results/full1m-spine-parents-json-recovery-20260911-r1`:

| Artifact | SHA-256 |
| --- | --- |
| `source-terminal.json` | `8b20fe37793f16a7c4d3290a8c562fa0b428914316af71ac2a54141b396d4378` |
| `saved-output-snapshots/population.json` | `f939bd0c4e52c8da796d3cb95333e7c4fa74358b4e7211c837946c16a57bd666` |
| `preflight.json` | `d7d6959f58fd451e8f39e084141d764206cb1fe5a9339a7986cca2cc92d8162f` |
| `input-snapshots/population.json` | `c1b6327b982e3ece01f2c26c6d46816067cdc6ab195da4cad630cf69ea603c04` |
| `preserved-completed-memories.json` | `c3e997a2da137db8c1d0646d10e721f1f62a712f21becdc93a025585b85be6a6` |
| `encoding-repair-verification.json` | `72116493340dca2e776e50f72e71028d2352afe768410967dc4315ccdaf8388c` |

## Active continuation and evaluation

Compiler session **14967**, PID **68916**, creation time
**1789132110.736258**, runs:

```powershell
.\.pixi\envs\dev\python.exe -X utf8 -m tools.run_local_spine_json_recovery compile --output-root eval_results/full1m-spine-parents-json-recovery-20260911-r1 --max-new-jobs-per-namespace 10000
```

The first two real four-summary batches completed with all outputs valid,
using the same NF4 generation model and FP16 compute as the previous compiler
(chunk `6ebf09`). Query-time Qwen attention, summary routing, raw hydration,
reader, judge and accuracy/latency gates are unchanged.

`tools/run_hierarchy_after_json_recovery.py` binds the new process and recovery
artifacts. Handoff session **87543** waits for that exact compiler, then requires
one complete bound ten-memory population and an idle workspace. It retains the
same two synthetic readiness calls, 400 fresh serial answer streams, 200 logical
judgments and zero-provider replay. Its preflight SHA is
`276a3346151fcb225a81925214ee22eaa764607b4d203fe9f0b5c60c160dd193`.

Handoff root:
`eval_results/full1m-hierarchy-after-json-recovery-20260911-r1`  
Evaluation root:
`eval_results/full1m-hierarchical-spine-joint-full100-20260911-r1`  
Source control:
`eval_results/full1m-spine-relative-reservation-full100-20260910-r1`

The handoff PID is **58468**, creation time **1789132187.0965445**. Its release
SHA is `db8c3524bf5fc205c0221c29ce8debb140cf573b8a9922a949d2c5a086f126bc`
(chunk `f2f0bd`).

The original compiler, parser, evaluator and handoff files remain frozen. The
successor files are now bound to live artifacts and must also remain unchanged.
Continue observing sessions 14967 and 87543; do not restart either live process.

## Fourth memory completed and replayed

Offset 030 completed all 473 source trees, 2,278 parents and 2,751 unchanged
original leaves after 107 new generation jobs across 35 successor batches
(chunk `5ef3db`). No dependencies remain. Its raw turns contain 1,043,571
`cl100k_base` tokens, as independently counted earlier.

An independent replay authenticated all 107 new response rows and reconstructed
the same complete hierarchy with generation forbidden. No model was loaded or
called, and every original leaf and raw-span descriptor remained unchanged.
All four complete memories use the same successor method policy. None of these
107 new rows required an encoding repair; the saved original failed response
had already been recovered in the input cache. Replay session 11636 exited 0
(chunk `8a787c`).

| Artifact | SHA-256 |
| --- | --- |
| Complete hierarchy | `4acb2691d51f5db032a6b65d602de1e44f8d5f1b2d85d7a0d62c8185c5edbd24` |
| Final progress | `fab9b46330e1e687fd329de06fd5e8d6845d66629efd6efd8547d996697dd502` |
| `fourth-namespace-verification.json` | `d74bf461fde1a2820f0f38c2c01bb5a82fcbedcf9546c53f0b7b61661e72ff6e` |

Four of ten memories are complete. The same compiler started offset 040 from
36 completed source trees and 18 parents, with 445 initial dependencies. Its
first batch is saved, and handoff session 87543 remains live (chunk `3a5b33`).

## Offset 040 first complete dependency wave

All 445 initial dependencies are resolved: 412 outputs passed immediately,
and all 33 oversized outputs passed their first bounded repair. The 121 batches
used 1,908.74 generation seconds at 24.27 aggregate tokens/s, with 5.183 GiB
peak allocation.

The compiler published 103 additional parents, reaching 81 complete source trees
and 121 parents, then began 400 further dependencies. Independent replay
authenticated every request/response and verified all 202 original leaves in
the completed trees unchanged, including raw-span descriptors. It made zero
model calls and read no raw transcript text (session 24589, exit 0, chunk
`3ff7c1`).

The verification is `offset040-first-wave-verification.json`, SHA
`5fd3cbe9be48bd105880aea3c7829f79cf135870fd744d9d4b6b94ebbce53603`.
The resulting progress SHA is
`ddd628472d681e6c749a454d28261272290ab42aed71b8ce0c8c0db7c92621c5`;
offset 040's preflight SHA is
`bf8ca7f3c9f803804625bd0e870f695b98eabf415fcedd2c8a02d3638777dd9e`.

A separate check examined encoding-repair metadata on all 478 response rows in
the same 121 authenticated batches. None required an encoding repair. Its receipt
is `offset040-first-wave-encoding-verification.json`, SHA
`4187d51c32fe653ae74f98d0e2b421c0797a0ed333d6195f41e00bf798008efd`
(chunk `3fbdfd`), with zero model calls. Both compiler session 14967 and handoff
session 87543 remained live at publication (chunks `f12526` and `ce0d86`). Four
memories are complete; the fifth and the full100 joint result remain pending.

## Offset 040 second complete dependency wave

All 400 dependencies from the first checkpoint are resolved: 364 outputs passed
immediately, and all 36 oversized outputs passed their first bounded repair.
The 109 batches used 1,746.37 generation seconds at 23.80 aggregate tokens/s,
with 5.179 GiB peak allocation. The compiler reached 914 total generation jobs
across 230 batches in this namespace, then published the checkpoint and started
352 further dependencies (chunk `219f8b`).

Independent verification authenticated all 436 response rows and checked all
401 original leaves in the completed trees unchanged, including raw-span
descriptors. The fifth memory now has 129/481 complete source trees and
272/2,256 parents, an increase of 151 published parents. The check read no raw
transcript text and made zero model calls (exit 0, chunk `6ab617`).

| Artifact | SHA-256 |
| --- | --- |
| Second-wave progress | `bb4c1f3274a1c82de20d7f21a8b3371813442132bc9d4d075f25021f8b1b7c26` |
| `offset040-second-wave-verification.json` | `66aeae65aab10c4efed915c907a510094cf188b32571fe0df3cf5770a8460015` |
| `offset040-second-wave-encoding-verification.json` | `fb1af1a17ce8e7070e81a58a0147ceec5f8a7ba37adbac3cc5dfc303e626378a` |

The separate encoding check verified metadata on the same 436 rows in 109
authenticated batches; none required an encoding repair (exit 0, chunk
`fbc5f6`). Compiler session 14967 and handoff session 87543 remained live after
checkpoint publication (chunks `4185e0` and `08978f`). Four whole memories are
complete; no hierarchical full100 answer or query-latency result is available.

## Offset 040 third complete dependency wave

All 352 dependencies from the second checkpoint are resolved: 320 outputs
passed immediately, and all 32 oversized outputs passed their first bounded
repair. The 96 batches used 1,531.90 generation seconds at 24.46 aggregate
tokens/s, with 5.163 GiB peak allocation. This brings the namespace to 1,298
generation jobs across 326 batches at the checkpoint.

Independent verification authenticated all 384 response rows and confirmed all
565 original leaves in the completed trees unchanged, including raw-span
descriptors. The fifth memory now has 164/481 complete source trees and
401/2,256 parents, an increase of 129 published parents. The check read no raw
transcript text and made zero model calls (exit 0, chunk `8d6d1b`).

| Artifact | SHA-256 |
| --- | --- |
| Third-wave progress | `644ab66d5920a5e147775f4118933c0c1ba43c6ab7e958869f8063dfbc0ac279` |
| `offset040-third-wave-verification.json` | `e63625a0903d41c364f09e84e3f21224eaab14a0194a765df21e7091d0991b8c` |
| `offset040-third-wave-encoding-verification.json` | `e26c16d689312558078637832efe2c4b3ba922b2244808220d206ebff32f6f26` |

The separate encoding check verified metadata on the same 384 rows in 96
authenticated batches; none required an encoding repair (exit 0, chunk
`65439a`). Compiler session 14967 published the checkpoint and started the next
317 dependencies (chunk `c4066c`); handoff session 87543 remained live (chunk
`040910`). Four whole memories are complete. The fifth memory and the full100
accuracy/latency result remain pending.

## Offset 040 fourth complete dependency wave

All 317 dependencies from the third checkpoint are resolved: 289 outputs passed
immediately, and all 28 oversized outputs passed their first bounded repair.
The 87 batches used 1,381.52 generation seconds at 24.88 aggregate tokens/s,
with 5.176 GiB peak allocation. This brings the namespace to 1,643 generation
jobs across 413 batches at the checkpoint.

Independent verification authenticated all 345 response rows and confirmed all
826 original leaves in the completed trees unchanged, including raw-span
descriptors. The fifth memory now has 206/481 complete source trees and
620/2,256 parents, an increase of 219 published parents. The check read no raw
transcript text and made zero model calls (session 48470, exit 0, chunk
`7dce90`).

| Artifact | SHA-256 |
| --- | --- |
| Fourth-wave progress | `f4f403ca00f58fd7854891626c01e38058992062e1a7841866a7d302d3447583` |
| `offset040-fourth-wave-verification.json` | `b2a94e4c9e12cde28d25a14eac791edde4e4e67179633d7ec8e8666f2bea5a68` |
| `offset040-fourth-wave-encoding-verification.json` | `18e3c970ab61aba7e37de67c39157d4ecd722e63742855370e90ad0a07eb0aac` |

The separate encoding check verified metadata on the same 345 rows in 87
authenticated batches; none required an encoding repair (exit 0, chunk
`916886`). Compiler session 14967 published the checkpoint with 275 next
dependencies (chunk `f63f48`); handoff session 87543 remained live (chunk
`1fc854`). Four whole memories are complete. The fifth memory and the full100
accuracy/latency result remain pending.

## Offset 040 fifth complete dependency wave

All 275 dependencies from the fourth checkpoint are resolved: 240 outputs passed
immediately, and all 35 oversized outputs passed their first bounded repair.
The 78 batches used 1,244.58 generation seconds at 25.07 aggregate tokens/s,
with 5.165 GiB peak allocation. This brings the namespace to 1,953 generation
jobs across 491 batches at the checkpoint.

Independent verification authenticated all 310 response rows and confirmed all
1,303 original leaves in the completed trees unchanged, including raw-span
descriptors. The fifth memory now has 277/481 complete source trees and
1,026/2,256 parents, an increase of 406 published parents. The check read no raw
transcript text and made zero model calls (exit 0, chunk `f8c5b4`).

| Artifact | SHA-256 |
| --- | --- |
| Fifth-wave progress | `74bcab163663b67d9a99154a2c075c4068ef8efa30b8117893fb6cd175868364` |
| `offset040-fifth-wave-verification.json` | `a1a122193102176f0b3423b96e34e30d90eaadd9640a55ff5aee885b59cf9f9f` |
| `offset040-fifth-wave-encoding-verification.json` | `a33e7e49aad207c871602728fa463207b633758b1a25e35b2a9c0764d543ceed` |

The separate encoding check verified metadata on the same 310 rows in 78
authenticated batches; none required an encoding repair (exit 0, chunk
`8057bc`). Compiler session 14967 published the checkpoint with 204 next
dependencies (chunk `a2b9fa`); handoff session 87543 remained live (chunk
`469250`). Four whole memories are complete. The fifth memory and the full100
accuracy/latency result remain pending.

## Offset 040 sixth complete dependency wave

All 204 dependencies from the fifth checkpoint are resolved: 182 outputs passed
immediately, and all 22 oversized outputs passed their first bounded repair.
The 57 batches used 929.63 generation seconds at 25.00 aggregate tokens/s,
with 5.164 GiB peak allocation. This brings the namespace to 2,179 generation
jobs across 548 batches at the checkpoint.

Independent verification authenticated all 226 response rows and confirmed all
1,867 original leaves in the completed trees unchanged, including raw-span
descriptors. The fifth memory now has 359/481 complete source trees and
1,508/2,256 parents, an increase of 482 published parents. The check read no raw
transcript text and made zero model calls (exit 0, chunk `54bc6e`).

| Artifact | SHA-256 |
| --- | --- |
| Sixth-wave progress | `c2de34b136ea3e370c80fb5f778afeadf914fd8d81f9bb8b236d2776b6f6675e` |
| `offset040-sixth-wave-verification.json` | `cfca130c762932a9510dcf12f63b6efaadf302633e8e587fbfade6261dc6a70f` |
| `offset040-sixth-wave-encoding-verification.json` | `0e2b9f629dddea26b17237d14a64094c900c15fa93fc77148ba650d8084f7611` |

The separate encoding check verified metadata on the same 226 rows in 57
authenticated batches; none required an encoding repair (exit 0, chunk
`e105a0`). Compiler session 14967 published the checkpoint and started the next
122 dependencies (chunk `079c7d`); handoff session 87543 remained live (chunk
`4afa1b`). Four whole memories are complete. The fifth memory and the full100
accuracy/latency result remain pending.

## Fifth memory completed and replayed

Offset 040 completed all 481 source trees, 2,256 parents and 2,737 unchanged
original leaves after 2,391 generation jobs across 605 batches (chunk `21dd21`).
No dependencies remain. Its raw turns contain 1,046,567 `cl100k_base` tokens,
as independently counted earlier.

The remaining waves after the sixth checkpoint were covered by an independent
replay of the whole completed namespace. It authenticated all 2,391 new response
rows, checked their encoding metadata and reconstructed the identical hierarchy
with generation forbidden. No model was loaded or called. Every original leaf
and raw-span descriptor remained unchanged, and the fifth memory uses the same
method policy as offset 000. None of these new response rows required an
encoding repair. Replay session 93965 exited 0 (chunk `0dad5c`).

| Artifact | SHA-256 |
| --- | --- |
| Complete hierarchy | `5ee96362f550caa7aef8743a85d30e4ab19b5983014980ae74460346dade3ba5` |
| Final progress | `aa00289ccd5dfede6b269e9713058a0e0758d3df5fcb279bef75d45161c0d7e4` |
| `fifth-namespace-verification.json` | `07007fd46a469e29df66c1c6207906a3128401e925f471f91856631bbc0abade` |

Five of ten memories are complete and independently replayed. Compiler session
14967 started offset 050 from 44 complete source trees and 17 parents, with 436
initial dependencies; its first two batches are saved (chunk `21dd21`). Handoff
session 87543 remains live and waits for all ten complete memories (chunk
`d606c7`). No new hierarchical full100 answer or query-latency result is claimed.

## Offset 050 first complete dependency wave

All 436 initial dependencies are resolved: 390 outputs passed immediately.
The remaining 46 outputs (45 over budget and one malformed JSON response)
all passed their first bounded repair. The 121 batches used 1,949.26 generation
seconds at 24.15 aggregate tokens/s, with 5.217 GiB peak allocation. The
checkpoint contains 482 generation attempts.

Independent verification authenticated every request/response and confirmed all
208 original leaves in the completed trees unchanged, including raw-span
descriptors. The sixth memory now has 88/480 complete source trees and
120/2,232 parents, an increase of 103 published parents. The check read no raw
transcript text and made zero model calls (exit 0, chunk `031761`).

| Artifact | SHA-256 |
| --- | --- |
| Offset 050 preflight | `c56f8611b7a03e0564eaa0f4147131344d442287abc5ac62b0a69f8059d21247` |
| First-wave progress | `02573022c7de2f7b1e7e71b5d4087bcae4dd962ebf6da1dfb9b76eeae389293b` |
| `offset050-first-wave-verification.json` | `d6ffddb9af084cd7adab425dc8dcd039086270dd0756ec6b5f43cc6116a29fbf` |
| `offset050-first-wave-encoding-verification.json` | `e35e4adddfb4cd1e4c02d009e0443056b3175d1b8e534dc299d589328b5f7f9f` |

The separate encoding check verified metadata on all 482 rows in the same 121
authenticated batches. None required an encoding repair (exit 0, chunk
`9221f4`). Compiler session 14967 started the next 392 dependencies (chunk
`16614b`); handoff session 87543 remained live (chunk `76e3e1`). Five whole
memories are complete. The sixth memory and the full100 accuracy/latency result
remain pending.

## Gateway availability rechecked at 16:45 UTC

A fresh, preflight-bound synthetic-summary request at 2026-09-11 16:45:28 UTC
still failed with HTTP 500 `InternalServerError`: the gateway reports the
`qwen3-8b-gguf` backend as an unknown model. This check was over six hours after
the preceding failed probe. It used a 30-second timeout, a 64-token output
limit and zero retries. One request and no completion response were saved;
no benchmark questions or raw corpus content were sent.

The probe root is `eval_results/qwen-gateway-readiness-20260911T1645Z`.
Its fixed single-request preflight SHA is
`c2182ad7e8cb5715cf5c7e5f1ec8d37e02dd9ad003a15cf9bbffa0b048861342`;
the observation SHA is
`84039820608ee46aaf61051eb6839d77c3ca9969304d1f8e73e4a8de4697f589`.
Probe session 79106 exited 0 after recording the failed observation (chunk
`ffd037`). No automatic retry or generator switch was performed. The local
compiler and handoff remained live (chunks `5f09ff` and `e8a552`); offset 050
was progressing through its second dependency wave.

## Offset 050 second complete dependency wave

All 392 dependencies from the first checkpoint are resolved: 357 outputs passed
immediately, and all 35 oversized outputs passed their first bounded repair.
The 107 batches used 1,688.88 generation seconds at 24.24 aggregate tokens/s,
with 5.161 GiB peak allocation. This brings the namespace to 909 generation
attempts across 228 batches at the checkpoint.

Independent verification authenticated every request/response and confirmed all
348 original leaves in the completed trees unchanged, including raw-span
descriptors. The sixth memory now has 125/480 complete source trees and
223/2,232 parents, an increase of 103 published parents. The check read no raw
transcript text and made zero model calls (exit 0, chunk `f77ce5`).

| Artifact | SHA-256 |
| --- | --- |
| Second-wave progress | `904c8f4c73d017cb1285349288157cf8842bc2d2615de253d7a24107f5c4a512` |
| `offset050-second-wave-verification.json` | `3c330ae9b6d07705a74b83068ec107252943e2e01cd049ee4c8b9235139273d3` |
| `offset050-second-wave-encoding-verification.json` | `379391aef3d7182b412188b859510bf141bd07e1db76c2772fc8b85fe227af66` |

The separate encoding check verified metadata on all 427 rows in the same 107
authenticated batches. None required an encoding repair (exit 0, chunk
`6c18e9`). Compiler session 14967 started the next 355 dependencies (chunk
`f0dd89`); handoff session 87543 remained live (chunk `b72e35`). Five whole
memories are complete. The sixth memory and the full100 accuracy/latency result
remain pending.

## Offset 050 third complete dependency wave

All 355 dependencies from the second checkpoint are resolved: 329 outputs
passed immediately, and all 26 oversized outputs passed their first bounded
repair. The 96 batches used 1,496.59 generation seconds at 24.44 aggregate
tokens/s, with 5.183 GiB peak allocation. This brings the namespace to 1,290
generation attempts across 324 batches at the checkpoint.

Independent verification authenticated every request/response and confirmed all
518 original leaves in the completed trees unchanged, including raw-span
descriptors. The sixth memory now has 158/480 complete source trees and
360/2,232 parents, an increase of 137 published parents. The check read no raw
transcript text and made zero model calls (exit 0, chunk `d2bb4e`).

| Artifact | SHA-256 |
| --- | --- |
| Third-wave progress | `ab6f677d80aec94ba02aa7540cdd0b00040908f173f27ce29071434c327199eb` |
| `offset050-third-wave-verification.json` | `9994d13810b027279c296f724b2b8896e50b49f09db55c1f949da222913c3267` |
| `offset050-third-wave-encoding-verification.json` | `48312f2f49e064ae2c9bce2df9bdb2b185b2b49fb87363fbe5e1a9e4723a6cfc` |

The separate encoding check verified metadata on all 381 rows in the same 96
authenticated batches. None required an encoding repair (exit 0, chunk
`64e7be`). Compiler session 14967 started the next 322 dependencies (chunk
`f8a622`); handoff session 87543 remained live (chunk `4ce73a`). Five whole
memories are complete. The sixth memory and the full100 accuracy/latency result
remain pending.

## Offset 050 fourth complete dependency wave

All 322 dependencies from the third checkpoint are resolved: 288 outputs passed
immediately, and all 34 oversized outputs passed their first bounded repair.
The 90 batches used 1,428.93 generation seconds at 24.55 aggregate tokens/s,
with 5.194 GiB peak allocation. This brings the namespace to 1,646 generation
attempts across 414 batches at the checkpoint.

Independent verification authenticated every request/response and confirmed all
780 original leaves in the completed trees unchanged, including raw-span
descriptors. The sixth memory now has 201/480 complete source trees and
579/2,232 parents, an increase of 219 published parents. The check read no raw
transcript text and made zero model calls (exit 0, chunk `f68ee5`).

| Artifact | SHA-256 |
| --- | --- |
| Fourth-wave progress | `a3c71b063328de1d65f78e5269f8518931935ef5e144a647aa884c25acf5c7d3` |
| `offset050-fourth-wave-verification.json` | `f02a3331ff1c9d0cf6c77de950fec822004027a3cdb69a8f976daa4634640b54` |
| `offset050-fourth-wave-encoding-verification.json` | `9e9998b836a886606c92a645879fc4f5b6ca4698aab2530f8fd217571ce54179` |

The separate encoding check verified metadata on all 356 rows in the same 90
authenticated batches. None required an encoding repair (exit 0, chunk
`1784af`). Compiler session 14967 published the checkpoint with 279 next
dependencies (chunk `f038e7`); handoff session 87543 remained live (chunk
`999d80`). Five whole memories are complete. The sixth memory and the full100
accuracy/latency result remain pending.

## Offset 050 fifth complete dependency wave

All 279 dependencies from the fourth checkpoint are resolved: 252 outputs
passed immediately, and all 27 oversized outputs passed their first bounded
repair. The 77 batches used 1,230.81 generation seconds at 25.13 aggregate
tokens/s, with 5.173 GiB peak allocation. This brings the namespace to 1,952
generation attempts across 491 batches at the checkpoint.

Independent verification authenticated every request/response and confirmed all
1,200 original leaves in the completed trees unchanged, including raw-span
descriptors. The sixth memory now has 265/480 complete source trees and
935/2,232 parents, an increase of 356 published parents. The check read no raw
transcript text and made zero model calls (exit 0, chunk `fc7be1`).

| Artifact | SHA-256 |
| --- | --- |
| Fifth-wave progress | `688319d981a4f430c6a69356c44cccc08fc5bc615bb56e6d076201063994eff1` |
| `offset050-fifth-wave-verification.json` | `24c5744fe999bc793283ca921d95501dbb741740062f0c7376dd21c91552dc2a` |
| `offset050-fifth-wave-encoding-verification.json` | `751ec8967bbafb066ef1b3d1632446dc1df08a4d813ecfdd12c5941c8057e2ab` |

The separate encoding check verified metadata on all 306 rows in the same 77
authenticated batches. None required an encoding repair (exit 0, chunk
`4c4bf1`). Compiler session 14967 started the next 215 dependencies (chunk
`8abf69`); handoff session 87543 remained live (chunk `f0b612`). Five whole
memories are complete. The sixth memory and the full100 accuracy/latency result
remain pending.

## Sixth memory completed and replayed

Offset 050 completed all 480 source trees, 2,232 parents and 2,712 unchanged
original leaves after 2,423 generation jobs across 613 batches (chunk `ad9826`).
No dependencies remain. Its raw turns contain 1,051,365 `cl100k_base` tokens,
as independently counted earlier.

The remaining waves after the fifth checkpoint were covered by an independent
replay of the whole completed namespace. It authenticated all 2,423 new response
rows, checked their encoding metadata and reconstructed the identical hierarchy
with generation forbidden. No model was loaded or called. Every original leaf
and raw-span descriptor remained unchanged, and the sixth memory uses the same
method policy as offset 000. None of these new response rows required an
encoding repair. Replay session 32941 exited 0 (chunk `b12c95`).

| Artifact | SHA-256 |
| --- | --- |
| Complete hierarchy | `24e55f397524dd3f93298ff26859f9dfe97326ce02e1baea15e783caf074ef51` |
| Final progress | `bdfcba463c428f3ead4f3e9b4da01b8ca17fa97eef0d404a3730bd9d8c668111` |
| `sixth-namespace-verification.json` | `13e331db7256a1f77e374e34c0bb790054d5310da660d0bfdf25624d7d269395` |

Six of ten memories are complete and independently replayed. Compiler session
14967 started offset 060 from 19 complete source trees and eight parents, with
438 initial dependencies; its first two batches are saved (chunk `ad9826`).
Handoff session 87543 remains live and waits for all ten complete memories
(chunk `bbfcea`). No new hierarchical full100 answer or query-latency result is
claimed.

## Offset 060 first complete dependency wave

All 438 initial dependencies are resolved: 395 outputs passed immediately, and
all 43 oversized outputs passed their first bounded repair. The 121 batches
used 1,881.38 generation seconds at 24.65 aggregate tokens/s, with 5.209 GiB peak
allocation. The checkpoint contains 68/457 complete source trees and 110/2,155
parents, with 389 next dependencies.

Independent verification authenticated all 481 response rows and confirmed all
178 original leaves in the completed trees unchanged, including raw-span
descriptors (exit 0, chunk `1043bb`). A separate check verified encoding metadata
on those same rows; none required an encoding repair (exit 0, chunk `f8eec1`).
Both checks made zero model calls and read no raw transcript text.

| Artifact | SHA-256 |
| --- | --- |
| First-wave progress | `7c699725960173b9063ab7eee4e7e0cf78d4bb144cd21b1d2a6573640cce0f58` |
| `offset060-first-wave-verification.json` | `184a51409a1a225f2974e16569bf8eb5401f5b29434abf134d82b066df072e05` |
| `offset060-first-wave-encoding-verification.json` | `9f93666d068fc9c252b2a67b8b6f8b4032056c4fce32da9e4382be74aaf75c05` |

Compiler session 14967 has started the next wave (chunk `525618`); handoff
session 87543 remains live (chunk `68b50f`). Six whole memories are complete.
The seventh memory and the full100 accuracy/latency result remain pending.

## Offset 060 second complete dependency wave

All 389 dependencies from the first checkpoint are resolved: 358 outputs passed
immediately, and all 31 oversized outputs passed their first bounded repair.
The 106 batches used 1,732.76 generation seconds at 23.00 aggregate tokens/s,
with 5.186 GiB peak allocation. This brings the namespace to 901 generation
attempts across 227 batches at the checkpoint.

Independent verification authenticated every request/response and confirmed all
358 original leaves in the completed trees unchanged, including raw-span
descriptors. The seventh memory now has 112/457 complete source trees and
246/2,155 parents, with 345 next dependencies. The check read no raw transcript
text and made zero model calls (exit 0, chunk `198270`).

| Artifact | SHA-256 |
| --- | --- |
| Second-wave progress | `e697b2a03fa1106aeea8b1f4c6ec0be1a1bed8fe0ad37a07ed82214e4d95d724` |
| `offset060-second-wave-verification.json` | `4a17dd9f72b61f232d6c994632d9d9e7ea8888782b148ce396828df48c59d766` |
| `offset060-second-wave-encoding-verification.json` | `d096f0ef365089bc9520d375d571b8e53759625f2b1d899941d8cdab8b84304f` |

The separate encoding check verified metadata on all 420 rows in the same 106
authenticated batches. None required an encoding repair (exit 0, chunk
`5c4c99`). Compiler session 14967 published the checkpoint (chunk `b538fa`);
handoff session 87543 remained live (chunk `96ac2d`). Six whole memories are
complete. The seventh memory and the full100 accuracy/latency result remain
pending.

## Seventh memory completed and replayed

Offset 060 completed all 457 source trees, 2,155 parents and 2,612 unchanged
original leaves after 2,310 generation jobs across 584 batches (chunk `60d8ea`).
No dependencies remain. Its raw turns contain 1,040,624 `cl100k_base` tokens,
as independently counted earlier.

The remaining waves after the second checkpoint were covered by an independent
replay of the whole completed namespace. It authenticated all 2,310 new response
rows, checked their encoding metadata and reconstructed the identical hierarchy
with generation forbidden. No model was loaded or called. Every original leaf
and raw-span descriptor remained unchanged, and the seventh memory uses the same
method policy as offset 000. None of these new response rows required an
encoding repair. Replay session 29577 exited 0 (chunk `0deb95`).

| Artifact | SHA-256 |
| --- | --- |
| Complete hierarchy | `dce4b46f1702f614460b10e5192cd9da75341c3f12cc48bb8e1eca231375f939` |
| Final progress | `583474666d73871e7f964fcb2db724e75cd7f048974d9c705eea349a2d29359d` |
| `seventh-namespace-verification.json` | `514b240f989057fd913dcd285bdfc1de917e52576178037483d9736c74a51ca3` |

Seven of ten memories are complete and independently replayed. Compiler session
14967 started offset 070 from 50 complete source trees and 18 parents, with 438
initial dependencies (chunk `60d8ea`); its first eight batches are saved (chunk
`f9f949`). Handoff session 87543 remains live and waits for all ten complete
memories (chunk `7ba346`). No new hierarchical full100 answer or query-latency
result is claimed.

## Qwen gateway recheck at 22:40 UTC

A fresh synthetic-only check at `2026-09-11T22:40:19.187923+00:00`, nearly six
hours after the previous check, still returned HTTP 500 with the unavailable
`qwen3-8b-gguf` backend error. It saved one request and zero completion responses
in 8.62 seconds, with a 30-second timeout, 64-token output cap and zero retries.
No raw corpus text or benchmark question was sent.

The observation is saved under
`eval_results/qwen-gateway-readiness-20260911T2240Z`, with SHA-256
`34e0575a1d4afa1873a6cd9ca51f8fa69bbc6103da419725583e97b06ace55ec`.
It reuses the fixed synthetic preflight
`c2182ad7e8cb5715cf5c7e5f1ec8d37e02dd9ad003a15cf9bbffa0b048861342`.
Probe session 82156 exited 0 after recording the failed availability check
(chunk `06d67a`); this is not a successful Qwen completion. Compiler session
14967 continued into the eighth memory (chunk `dcfd97`), and handoff session
87543 remained live (chunk `849035`). The generation backend was not switched.

## Eighth memory completed and replayed

Offset 070 completed all 488 source trees, 2,198 parents and 2,686 unchanged
original leaves after 2,321 generation jobs across 597 batches (chunk `4adf95`).
No dependencies remain. The earlier independent complete-turn count is
1,039,792 `cl100k_base` tokens; the inherited `raw_token_proxy` remains
1,039,791. The existing one-token proxy difference was preserved.

An independent replay authenticated all 2,321 new response rows and checked
their encoding metadata. Two rows used the existing narrow apostrophe-escape
repair, and both repairs verified against their saved original outputs. The
whole hierarchy reconstructed identically with generation forbidden. No model
was loaded or called. Every original leaf and raw-span descriptor remained
unchanged, and the eighth memory uses the same method policy as offset 000.
Replay session 10366 exited 0 (chunk `a5e031`).

| Artifact | SHA-256 |
| --- | --- |
| Complete hierarchy | `8a18d17515e60aeea49b243a79aeb510a817189df4a483d45dc29cf3d369e119` |
| Final progress | `651836a660a215395f71603a3a1805486ef572d650f8ce9d7fc794697be18046` |
| `eighth-namespace-verification.json` | `f406e75ed64101a779b0e9f50af11a3e772ca268d266cbd2f671eea7bcf53e67` |

Eight of ten memories are complete and independently replayed. Compiler session
14967 started offset 080 from 35 complete source trees and 13 parents, with 453
initial dependencies (chunk `4adf95`); its first five batches are saved (chunk
`26da0c`). Handoff session 87543 remains live and waits for all ten complete
memories (chunk `091712`). No new hierarchical full100 answer or query-latency
result is claimed.

## Ninth memory completed and replayed

Offset 080 completed all 488 source trees, 2,220 parents and 2,708 unchanged
original leaves after 2,367 generation jobs across 600 batches (chunk `8a043e`).
No dependencies remain. Its raw turns contain 1,041,987 `cl100k_base` tokens,
as independently counted earlier.

An independent replay authenticated all 2,367 new response rows and checked
their encoding metadata. None required an encoding repair. The whole hierarchy
reconstructed identically with generation forbidden; no model was loaded or
called. Every original leaf and raw-span descriptor remained unchanged, and
the ninth memory uses the same method policy as offset 000. Replay session
77492 exited 0 (chunk `dc6caa`).

| Artifact | SHA-256 |
| --- | --- |
| Complete hierarchy | `ac4f19e1e4e7c95402422c9109f9c3de575533e8c6d53687dc957d7e23b9096f` |
| Final progress | `76f5576e13bcd0b63c87161068e1eacdb279bb357c96482640106ffcd834e934` |
| `ninth-namespace-verification.json` | `4c386aab8a5e3f0f18da6889a33a08f9b2919ad54139bc6ce39e723821f2173c` |

Nine of ten memories are complete and independently replayed. Compiler session
14967 started offset 090 from 49 complete source trees and 30 parents, with 447
initial dependencies (chunk `8a043e`); its first five batches are saved (chunk
`a01a50`). Handoff session 87543 remains live and waits for all ten complete
memories (chunk `93acca`). The full100 accuracy and query-latency result is
still pending.

After the tenth memory finishes, let the handoff own the idle evaluation
workspace. Its population admission checks run before prompt preparation;
perform the separate tenth-memory replay after the timed evaluation exits.
Starting another Python verifier during the transition could trip the
evaluator's idle-workspace check or overlap the timed comparison.

## Tenth memory completed; evaluation handoff live

Offset 090 completed all 496 source trees, 2,298 parents and 2,794 unchanged
original leaves after 2,408 generation jobs across 621 batches. Its raw turns
contain 1,046,567 `cl100k_base` tokens, as independently counted earlier.
Compiler session 14967 exited 0 (chunk `6049c9`) after publishing the complete
ten-memory population. Across all ten memories, the hierarchies contain 4,805
source trees, 22,257 parents and 27,062 original leaves.

| Artifact | SHA-256 |
| --- | --- |
| Offset 090 hierarchy | `e87297f6b4de99aa10b89003d34e55cd46b3d51b0a8668e1ff8e3058745d385a` |
| Offset 090 final progress | `577be0dd05babe26c18ff49ac48f6d12ed414a674428d5f67240ba47c68a14ef` |
| Complete parent population | `7adbeefc326fe9180966d0fb10b98d1e19f435a11a473872e119fc020417981e` |

The population is saved at
`parents/populations/880cab563f214b64334fc36381cad087b9551c2134cb2ec72cdf5c1c01bce77e.json`
under the recovery root. Handoff session 87543 remains live (chunk `757cdf`).
The separate tenth-memory replay is deferred until the timed evaluation exits,
as described above. Compilation completion is not an answer-accuracy or
query-latency result; the full100 comparison remains pending.

## Tenth replay and full100 evaluation completed

After the timed evaluation exited, independent replay authenticated all 2,408
new offset-090 generation rows across 621 batches. None required an encoding
repair. Generation was forbidden and no model was loaded; the reconstructed
hierarchy, original leaves and raw-span descriptors were identical. Replay
session 52658 exited 0 (chunk `1f769a`). The receipt
`tenth-namespace-verification.json` has SHA-256
`fa8683dba410ccba3609350e958407672589a71e71b2743b172620ef27d43a74`.
All ten hierarchies are now independently replayed.

Handoff session 87543 exited 0 after 400 fresh streamed answers, 200 logical
Sol judgments (196 physical calls), and an identical zero-call report replay.
The hierarchy scored **8/100**, versus **84/100** for the flat control.
Its median total latency was **6.46 s**, versus **4.49 s** for the same-evidence
API control. Both gates failed. Successful compilation and replay certify
artifact integrity, not useful retrieval. The saved-route failure assessment
and complete timings are in [Research Log 181](181%20-%202026-09-12%20-%20Full100%20hierarchy%20failure%20assessment.md).

## Validation and limits

48 focused checks passed in 5.19 seconds (chunk `24b098`), covering encoding
boundaries, replay, no-EOS and token-budget rejection, unacknowledged requests,
cache preservation, original bounded generation recovery, and handoff ordering.
The real failed summary and completed-memory preservation were verified with
zero model calls. All ten memories are complete and independently replayed.
The fresh joint evaluation rejects the hierarchy; the latest measured flat
control is 84/100. The 95% joint target remains unproven.
