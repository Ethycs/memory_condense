# Four-summary local Qwen ingest batches

Four independent summary sequences completed in 13.64 seconds, compared with
24.82 seconds for the same four jobs in the earlier two-sequence batches. All
four outputs reached EOS and passed the original summary parser and token
budget. Peak GPU allocation was 5.072 GiB. This measured 1.82x wall-time speedup
applies to this ingest sample, not to complete compilation or query latency.

The successor is now compiling all ten memories. It reuses 90 valid local
summaries in addition to the previously authenticated gateway summaries.
Offset 000 starts from 159 complete source trees and 345 parents, compared with
144 source trees and 274 parents before local generation. No complete namespace
or new answer-accuracy result is available yet.

## Depth admission

All 4,805 saved source trees were reconstructed without new Qwen calls or raw
text reads. Maximum tree depths by offset are 8, 7, 7, 8, 8, 8, 7, 9, 8 and 10.
Every tree fits the query router's limit of 16 levels.

The sealed diagnostic is
`eval_results/full1m-spine-parent-population-20260910-r1/query-depth-admission.json`,
SHA `53ce0bfc1686c6a2502da5b37ebdb6ebb6ab3658c878f6dfe2a578455b34eefc`.
Session 17181 completed with exit zero. This checks structural admission only;
it does not measure selection quality or traversal latency.

## Batch policy and completed-output reuse

`tools/run_local_spine_parent_batch4.py` retains the same pinned full Qwen model,
NF4/FP16 compute configuration, CPU embeddings, independent summary messages,
generation cap and validation. It changes the maximum independent batch size
from two to four and binds its source hash into the backend identity.

Its measured release requires all four bound outputs to be valid, total time
below the original two-batch baseline, and peak allocation below 5.3 GiB.
The original real request/response journals supply the comparison jobs. No
question gold or raw support text enters this benchmark.

`tools/reuse_local_spine_parent_summaries.py` creates a separate frozen summary
snapshot. It authenticates the original cache and local preflights, source
implementation hashes, corpus bindings, exact typed messages, backend identity,
response bindings and attribution. It imports only outputs accepted by the
existing parser. Conflicting summaries fail. Invalid or uncompleted outputs
remain excluded and their original artifacts are preserved.

Ten tests pass in 2.09 seconds (chunk `bb85c9`). They cover completed-output
reuse and replay, invalid/incomplete exclusion, wrong corpus/raw/job bindings,
unchanged original-cache reuse, source-root protection, and the measured batch
release's validity, speed and memory conditions. `git diff --check` passes.

## Execution state

The prior two-sequence compiler was stopped deliberately to free the GPU for
this measured policy change. This was not an observation timeout. PID 57600
and creation time 1789097451.244538 were verified before termination; terminal
state was then observed. Session 1144 returned exit one; its Python process
reported termination code 15. All completed artifacts remain in the old root.
The terminal receipt is
`eval_results/full1m-spine-parents-local-20260910-r1/batch-policy-terminal.json`,
SHA `4516b38fef33bda1e5de8c749936aad37d0d7703a6d36585b432e21596053560`.

The active output root is
`eval_results/full1m-spine-parents-local-batch4-20260910-r1`.
The measured batch result is `batch-result.json`, SHA
`84970510868adbd38203c1b84bcff70a0e413295c87c8d6b1ccafbdd65b08433`.
The frozen input population is `input-snapshots/population.json`, SHA
`64312e3190d32d7b94bb25d07cff7560a0f52312c2304618a53ed537a8a7ade0`.
It includes the 90 additional valid local summaries and excludes two request
journals without completed responses. An excluded prepared request is not
necessarily an interrupted model call.

The snapshot was independently replayed with zero model calls and the same
population SHA (chunk `6acc74`). The first six subsequent four-sequence batches
completed at 21.45–29.54 generated tokens/s and at most 5.148 GiB peak allocation.
Nineteen of those 24 outputs passed immediately; five remain subject to the
existing bounded recovery policy. These are validation counts, not answer
accuracy or a claim about semantic completeness of the summaries.

Session **65191 is active**. It passed the batch release, created all ten input
snapshots and entered offset-000 compilation. Parent outputs are under
`parents/offset-NNN`. The same resident model proceeds through all ten memories,
with a 10,000-new-job allowance per namespace including bounded recoveries.
Poll this handle; do not restart a live execution because a tool observation
yields or times out. The batch benchmark's reservation is intentionally
single-use, so a later compiler resume must use the frozen input population
and authenticated `BatchFourQwen` backend rather than re-running that benchmark.

The 95/100 plus API-like latency target remains active and unmet. The next
quality result must come from the complete, fresh full100 joint evaluation.

## First complete dependency wave

The first offset-000 wave resolved all 340 initially missing merge jobs.
Initially 311 outputs passed and 29 exceeded the summary token budget. All 29
passed their first bounded recovery attempt; no second recovery was required.
The 93 completed batches used 1,374.78 generation seconds at 25.07 aggregate
generated tokens/s, with peak GPU allocation 5.213 GiB.

The compiler advanced from 159 complete source trees / 345 parents to 212
source trees / 579 parents, publishing 234 additional parents. It then began
the next wave of 287 dependencies. The memory remains incomplete overall.

An independent verification replay authenticated every first-wave request and
response with zero new model calls. It confirmed that all 340 dependencies
are accepted and that every original leaf in the 212 published source trees
is unchanged: 791 leaves, with exact original summary and raw-span descriptors.
This verifies the published subset, not completion of the whole memory.

The verification artifact is `first-wave-verification.json`, SHA
`a922f439ddb02d0de04cf1c9091d905a99bc23ad4060add4563678e906605bd2`
(tool chunk `f856e2`). The resulting progress SHA is
`a7e9887c3f8829c838dd9ce9281107c96bcb6e23537c552656056edf135d89f9`.
Sessions 65191 (compiler) and 61096 (evaluation handoff) remain active. These
summary-validation results do not measure answer accuracy.

## Second complete dependency wave

The second wave resolved all 287 dependencies. Initially 266 outputs passed
and 21 exceeded the summary budget. Twenty passed their first recovery; the
remaining output passed its second bounded recovery. Across 79 batches, the
wave used 1,179.78 generation seconds at 25.23 aggregate tokens/s, with peak GPU
allocation 5.182 GiB. No recovery exhausted the configured allowance.

The compiler published another 317 parents, advancing offset 000 to 272
complete source trees and 896 parents, then began 227 further dependencies.
Independent replay made zero model calls and authenticated all second-wave
requests/responses and every original leaf in those complete trees: 1,168
unchanged leaf descriptors, including their exact raw spans. The full memory
is still incomplete.

The verification artifact is `second-wave-verification.json`, SHA
`864b41e4a94cc7ac99cd5558fc2f944e23b57747e3658afdfde982ce4c57b588`
(tool chunk `1e42dd`). The resulting progress SHA is
`fa07853690fddb6105b2f7f7fe39c4f4c6b5901da48691fff974a73c4134af8f`.
The compiler and evaluation handoff remain live on sessions 65191 and 61096.

An optional model-role clarification is open: whether Terra may generate parent
summaries while Qwen handles attention routing. No reply has been received.
The current Qwen generator and scheduled evaluation continue under the existing
method; neither is waiting for that clarification.

## Third complete dependency wave

All 227 third-wave dependencies are resolved. Initially 205 outputs passed
and 22 exceeded the summary budget; all 22 passed their first bounded recovery.
The 63 batches used 944.74 generation seconds at 26.32 aggregate tokens/s and
5.183 GiB peak allocation. The compiler published 343 more parents, reaching
334 complete source trees and 1,239 parents, then started 165 further dependencies.

`tools/verify_local_parent_wave.py` consolidates the read-only replay used for
these wave checks. It authenticates typed request/response bindings and accepted
summary outputs, then verifies the unchanged original leaves in every published
source tree. It makes no model calls and reads no raw transcript text. Running
it on the first two waves reproduced both existing verification artifacts
byte for byte (chunk `de69e8`), and it verified the third wave's 1,573 unchanged
leaf descriptors (chunk `d352b6`).

The third verification is `third-wave-verification.json`, SHA
`aa34f0d0afbdd0a030b4ec3c6c93bb4f3775e20e827cb29ea0c7e253eb1408a9`.
The resulting progress SHA is
`7100f352134ba0dad05b65864164db447bea7f3865b10116d1c42214762321e9`.
The complete namespace and all-ten-memory evaluation are still pending.

## Fourth complete dependency wave

All 165 fourth-wave dependencies are resolved. Initially 152 outputs passed
and 13 exceeded the summary budget; all 13 passed their first bounded recovery.
The 46 batches used 669.95 generation seconds at 26.56 aggregate tokens/s and
5.171 GiB peak allocation. The compiler published another 364 parents, reaching
397 complete source trees and 1,603 parents, then began 102 further dependencies.

Independent replay verified all requests/responses and all 2,000 unchanged
original leaf descriptors in the completed trees, with zero model calls and
zero raw-text reads. The verification is `fourth-wave-verification.json`, SHA
`76c93c2a335e382549ed1bdb96334c4b3232718292fe4af8931a477fb08befd6`
(tool chunk `0a8dfe`). The resulting progress SHA is
`bdb947d4bb0ca6ddfeec314da91a7b691b3844867fb375dac665ac5751f536b9`.
The full offset-000 hierarchy and full100 evaluation remain pending.

## First complete memory and independent replay

Offset 000 is now complete: all 499 source trees, 2,245 planned parents and
2,744 original leaves, with raw-token proxy 1,041,276. Its final progress SHA is
`4075f1227b1df7a2494bd009187b7ca357b2894919a5acf47cca1368e2baf9c2`;
the complete `parents/offset-000/hierarchy.json` SHA is
`91b09d9e469f087d19e5b329b72f788cec53b2a7e7ef5c6ff03b5fd5ae4e1f0d`.
The compiler used 1,302 new local generation jobs across 334 batches in this
namespace, including bounded repairs, then began offset 010 on the same
resident model (chunk `ff4c2f`). No remote provider calls were used.

The fifth and sixth waves were separately verified with the existing replay
utility. They resolved 102 and 56 dependencies respectively, reaching 443 and
480 complete source trees. The verification SHAs are
`76c58eec4384af6dded5468e6a43575c20a06a42d27f9d16f89daab47a6e5a7d`
and `e36dca4fcce5127b7866b2df6276dae36474e7f693153b901057f9ddb439ea45`
(chunks `bdee36` and `c911e3`). Subsequent waves resolved the final 19 source
trees and published the complete hierarchy.

An independent zero-generation replay ran the existing compiler over the
completed namespace, with generation explicitly forbidden and a zero-job
allowance. It authenticated the saved requests/responses, reconstructed every
source tree from the original plans and verified exact preservation of all
2,744 leaf descriptors, including raw-span references. The resulting hierarchy
has the identical SHA; the model was never loaded, and no model or provider
calls were made. The verification artifact is
`first-namespace-verification.json`, SHA
`c9f89b09b91ae0067520b08056b1455c796ed9f5976f5b021380012139b61433`
(chunk `4806f5`).

Sessions 65191 and 61096 remain live: the compiler is generating the second
memory and the scheduled full100 evaluation still requires all ten. This is
one completed ingestion, not a new answer-accuracy or query-latency result.

A read-only cross-memory check authenticated all ten original leaf indexes and
compared source identities and complete leaf-descriptor populations. All 4,805
source IDs and all 4,805 source leaf-tree fingerprints are distinct (chunk
`4e03d1`), so no complete source tree can be reused unchanged between namespaces.
This rules out whole-tree copying; it does not rule out coincidentally identical
individual summary requests. The compiler and evaluation method are unchanged.

## Offset 010 first complete dependency wave

The second memory advanced from 46 complete source trees / 32 parents to
83 source trees / 113 parents, then started 381 further dependencies. All
418 initial dependencies are resolved: 383 outputs passed immediately, 34
exceeded their summary budgets and one was not JSON. All 35 passed their first
bounded repair. The 114 batches used 1,632.31 generation seconds at 26.98
aggregate tokens/s and 5.171 GiB peak allocation.

Independent replay authenticated those requests and responses and verified all
196 original leaves in the published trees unchanged, including raw-span
descriptors, with zero model calls and zero raw-text reads. The verification is
`offset010-first-wave-verification.json`, SHA
`2120cb9ad293d60e836dc7bff3701bf82326c5bcb0e316733b0aa521b2996971`
(chunk `7aac68`). The resulting progress SHA is
`a01e236afe76dddbc5c66db3b8430b31bb636cb4a40efc1d24c8e552414e11ab`;
offset-010 preflight SHA is
`dc573e147b95adab655b5c177c050e46f967b41d28df15f001df5d03f114d483`.

One of ten memories is complete. Sessions 65191 (compiler) and 61096 (full100
handoff) remain live. No new answer-accuracy or latency result is claimed.

At 2026-09-11 05:44 UTC, a fresh gateway check still found no working Qwen
alternative. `/v1/models` returned 13 model IDs with only `qwen3-8b` matching
Qwen; one synthetic readiness request returned HTTP 500 for the same unknown
`qwen3-8b-gguf` backend. No user content was sent and automatic retries were
disabled. The model-list and readiness artifacts are under
`eval_results/qwen-gateway-model-discovery/`, with respective SHAs
`8a9d8838dad4656f20cc30d3c52ab15b812773d968f5a1f704823056db689e69`
and `3cad49afa4b76f383b1b54f79320abff3f8c5af8a737338b60fc171a5e6ba650`
(chunks `4c1633` and `05b090`). The active local compiler was not interrupted.

## Whole-turn memory-size recount

A separate read-only recount authenticated all ten serving-index bindings,
raw-store hashes, unique turn IDs and individual raw-text hashes, then counted
the complete turns with the repository's `cl100k_base` BPE tokenizer. Every
memory exceeds one million tokens without chat framing. Counts by offset are
1,041,276; 1,044,341; 1,045,527; 1,043,571; 1,046,567; 1,051,365; 1,040,624;
1,039,792; 1,041,987; and 1,046,567. The minimum is **1,039,792**.

These are actual counts under the named public encoding, which remains a proxy
for the answer provider's tokenizer. Fragment and whole-turn tokenization can
differ; every recomputed whole-turn count matched the existing whole-turn
accounting where present, or the original compiler's whole-turn count otherwise.
No model or provider calls were made, and no raw inputs went to Qwen.

`whole-turn-size-verification.json` binds all ten input artifacts and the exact
tokenizer vocabulary and version. Its SHA is
`d86be5b2f85f6a5c4169a7923b90818d449244ab038054f0d17f8d1c50a1bc5c`
(chunk `b42bbe`). This supports the benchmark's memory-size requirement only;
the full100 accuracy and latency target remains unproven.

## Offset 010 second complete dependency wave

All 381 dependencies in the second wave are resolved. Initially 334 summaries
passed; 47 required repair. Forty-six passed their first repair and the remaining
output passed its second. The 109 batches used 1,656.19 generation seconds at
24.91 aggregate tokens/s, with 5.190 GiB peak allocation.

The compiler published 129 additional parents, reaching 126 complete source
trees and 242 parents, then began 338 further dependencies. Independent replay
authenticated every request/response in this wave and verified all 368 original
leaves in the completed trees unchanged, including exact raw-span descriptors.
It made zero model calls and read no raw transcript text.

The verification is `offset010-second-wave-verification.json`, SHA
`f75adb7a34f801c58da3068d2a45bdb3318e9e5d22da5c8ecb1fe1bfb58e5409`
(chunk `38a0e5`). The resulting progress SHA is
`e291637ec170b04ebfdb04cffdd7c2f7e1e36161c636132a61d0e23cef4f6986`.
The second memory and full100 evaluation remain unfinished; both scheduled
processes continue on sessions 65191 and 61096.

## Offset 010 third complete dependency wave

All 338 third-wave dependencies are resolved: 306 outputs passed immediately
and all 32 oversized outputs passed their first bounded repair. The 93 batches
used 1,435.54 generation seconds at 24.93 aggregate tokens/s, with 5.175 GiB
peak allocation. The compiler published another 80 parents, reaching 147
complete source trees / 322 parents, then began 317 further dependencies.

Independent replay authenticated every request/response and verified all 469
original leaves in the published trees unchanged, including raw-span descriptors,
with zero model calls and zero raw-text reads. The verification artifact is
`offset010-third-wave-verification.json`, SHA
`ee83f6c8f2fdff25234354895242a1c8fe2761e346791ace8ee3d216e668713e`
(chunk `f62c63`). The resulting progress SHA is
`6745527e23540ef6f57cbde4b9a892e7ad2c17355f434bdfe3e3fcccf5e9c86c`.
One complete memory and this verified subset do not satisfy the full100 target.
The compiler and evaluation handoff remain live.

## Offset 010 fourth complete dependency wave

All 317 fourth-wave dependencies are resolved: 281 outputs passed immediately
and all 36 oversized outputs passed their first bounded repair. The 89 batches
used 1,393.89 generation seconds at 25.10 aggregate tokens/s, with 5.164 GiB peak
allocation. The compiler published 213 more parents, reaching 190 complete
source trees / 535 parents, then began 274 further dependencies.

Independent replay authenticated every request/response and verified all 725
original leaves in the published trees unchanged, including raw-span descriptors.
It made zero model calls and read no raw transcript text. The verification is
`offset010-fourth-wave-verification.json`, SHA
`9fc157f33088c9a4b1c00b8dfdf74f19b4b6dc2b44c5e34661fc93c8a44ccd4e`
(chunk `27d7dc`). The resulting progress SHA is
`27c011ace84326c4fdafd0643db36199695a065f1655eebbf36f7c5cc5add39f`.
The second memory and full100 evaluation remain incomplete; the compiler and
handoff continue on sessions 65191 and 61096.

## Offset 010 fifth complete dependency wave

All 274 fifth-wave dependencies are resolved. Initially 245 outputs passed;
all 29 oversized outputs passed their first bounded repair. The 77 batches used
1,205.26 generation seconds at 25.53 aggregate tokens/s, with 5.171 GiB peak
allocation. The compiler published 420 additional parents, reaching 263
complete source trees / 955 parents, then began 201 further dependencies.

Independent replay authenticated all requests/responses and verified all 1,218
original leaves in the completed trees unchanged, including their exact raw-span
descriptors. It made zero model calls and read no raw transcript text. The
verification is `offset010-fifth-wave-verification.json`, SHA
`57f0fbcdab91810ad34cc91f8bca2f6ddcd6d735520d94ead9468a2e04399ab6`
(chunk `66ea8a`). The resulting progress SHA is
`2820a1484251f169fa39df97bec2a4313b37edb4005cc24bdf38734d848fded5`.
The compiler and full100 handoff remain live; the second complete memory and
joint accuracy/latency result are still pending.

## Offset 010 sixth complete dependency wave

All 201 sixth-wave dependencies are resolved: 176 outputs passed immediately
and all 25 oversized outputs passed their first bounded repair. The 58 batches
used 931.26 generation seconds at 25.11 aggregate tokens/s, with 5.171 GiB peak
allocation. The compiler published 499 additional parents, reaching 349
complete source trees / 1,454 parents, then began 115 further dependencies.

Independent replay authenticated every request/response and verified all 1,803
original leaves in the published trees unchanged, including raw-span descriptors.
It made zero model calls and read no raw transcript text. The verification is
`offset010-sixth-wave-verification.json`, SHA
`f7a8e74790af9c7b248ff312e12dc6b6aafbd27ac0b0da3a36f811cbce2c88bb`
(chunk `1af5db`). The resulting progress SHA is
`1b45bc8d61ff315575457d4cab69a80f2f6a68d614a1e9cb23324768c2d2f243`.
Both compilation and the full100 handoff remain live; no new answer score or
query-latency result is available.

The existing Sol reader diagnostic from Log 167 was also rechecked against its
sealed preflight, answers, report and completion artifact (chunk `620a6a`),
with zero new model calls. It judged only 4/20 prior misses correct, with the
previously documented grading inconsistencies, and did not reanswer the other
80 questions. It does not establish a stronger-reader solution or a full100
score; no duplicate reader campaign was started during this compilation wait.

## Second memory complete and replayed — 2026-09-11

Offset 010 is complete: all 464 source trees, 2,161 parents and 2,625 original
leaves over 1,044,341 whole-turn `cl100k_base` tokens. Its final progress SHA is
`0754fdc10239f9c006278c7486157ecdd2431bdce518cd860df6d80bfb96d9cf`;
the complete hierarchy SHA is
`1fd4930d22226b1bb9946c6e3042c350e5d70a2c4f260bc5110463540d6282ef`.

The seventh and eighth dependency waves have independent verification artifacts:

| Artifact | SHA-256 |
| --- | --- |
| `offset010-seventh-wave-verification.json` | `a0a7947c7b0f332a24baee66f038649e2fb52607a78e4431cf529d12f633406d` |
| `offset010-eighth-wave-verification.json` | `c72983b9c308c2de5f1b6a49476e1d1206349d2f71ec27792ce5da330102c8a0` |

After the remaining dependencies completed, an independent replay reconstructed
the entire namespace with generation forbidden and a zero-job allowance. It
produced the identical hierarchy, preserved the original leaves and raw-span
references, and confirmed the same method policy as offset 000. The model was
never loaded; new local and remote calls were both zero. The receipt is
`second-namespace-verification.json`, SHA
`2d79952168757e4535d304e7cb96873cb15e59a7e9127cd222f13e21de1b7dca`.
Its hierarchy binding and replay invariants were checked again in chunk
`ac45f6` without model calls.

Two of ten memories are now complete. The original compiler session 65191 has
started offset 020, and evaluation handoff session 61096 is still waiting for
that exact process to finish all ten memories. Both handles were confirmed live
in chunks `3a5822` and `b2686c`. These are ingestion and reconstruction results;
no new answer-accuracy or query-latency result is available.

### Manual summary-fidelity sample

A manual comparison inspected eight saved parent outputs against their supplied
child summaries: four attached-context jobs from offset 020 and four user-spine
jobs from offset 010. Selection used request filename order and job kind, not
benchmark questions or answer correctness. All eight outputs passed their summary
contract. No model calls were made and no raw transcript text was read.

**Known semantic loss:** one user parent reduced conflicting child statements
about a completed versus planned solo drive to Yosemite to a single planned
drive. It also omitted the planned sunrise at Tunnel View. An attached-context
parent preserved the curry/naan topics and assistant attribution but omitted
three named curry-powder brands. Another user parent retained the vase's $800
price, the watch's grandfather inheritance, and the purported first-edition
qualifier. These are individual observations, not a corpus-wide fidelity score.

The original leaves and raw evidence remain intact. This check identifies ways
parent summaries could lose routing signals; it does not show whether the full100
queries are affected. Complete the unchanged full evaluation, then inspect these
mechanisms in any measured routing losses before selecting a successor method.

`parent-summary-semantic-spot-check.json` binds the two request/response pairs,
all eight jobs, selection rules, findings and limitations. Its SHA is
`9ce71f89069fd2b1a1b1ec00607a4dbbd7d605b9f079ecb267e8ae54c00e19ae`
(chunk `f9420f`).

### Remaining build time and exact-prompt reuse check

At 2026-09-11 08:03 UTC, offset 010's 604 saved batches accounted for 2.562
generation hours and 2.583 hours from namespace preflight to complete hierarchy.
Offset 020 had 81 saved batches and 0.369 elapsed hours (chunk `d4345d`). At
comparable per-memory throughput, roughly 20 additional compilation hours
remained, plus preparation and evaluation. This is a planning estimate, not a
completion deadline or a query-latency measurement.

A read-only scan of saved local responses found 1,204 distinct original prompt
IDs in offset 000 and 2,137 in offset 010, with zero intersection (chunk
`0e0858`). Prompt IDs hash the actual canonical summary messages, without source
IDs or raw-span descriptors. Thus the generated work in these two completed
memories offers no exact-message reuse between them. This bounded check excludes
imported caches and future jobs; it does not rule out every later reuse opportunity.

## Offset 020 first complete dependency wave

All 443 initial dependencies are resolved: 399 outputs passed immediately,
43 exceeded their summary budgets and one did not reach EOS. All 44 rejected
outputs passed their first bounded repair. The 122 batches used 1,908.34
generation seconds at 24.67 aggregate tokens/s, with 5.184 GiB peak allocation.

The compiler published 126 additional parents, reaching 91 complete source trees
and 149 parents, then began 388 further dependencies. Independent replay
authenticated the saved requests/responses and verified all 240 original leaves
in the completed trees unchanged, including their raw-span descriptors. It made
zero model calls and read no raw transcript text.

The verification is `offset020-first-wave-verification.json`, SHA
`a8afffb01919e1f0902a30c085002af85569efd16b6f8156845ebfb8fd992a16`
(chunk `e60656`). The resulting progress SHA is
`b122b9df60ef194795fffa346691a295fc0e5331e47e2c845d33eb009aa0a253`;
offset 020's preflight SHA is
`e360fccbb7843fb9326e08082ef27ecc4cd4ecedd628b010421105373fdba93b`.

Two of ten memories are complete. Sessions 65191 and 61096 remain live; the
third memory and full100 accuracy/latency result are still pending.

## Offset 020 second complete dependency wave

All 388 second-wave dependencies are resolved: 351 outputs passed immediately,
and all 37 oversized outputs passed their first bounded repair. The 107 batches
used 1,635.69 generation seconds at 25.35 aggregate tokens/s, with 5.207 GiB
peak allocation.

The compiler published 105 additional parents, reaching 126 complete source trees
and 254 parents, then began 353 further dependencies. Independent replay
authenticated every request/response in the wave and verified all 380 original
leaves in the completed trees unchanged, including raw-span descriptors. It made
zero model calls and read no raw transcript text.

The verification is `offset020-second-wave-verification.json`, SHA
`a5c711a9ab20c6cf2e8997900ed1da454d6e9438e7e67c1b38a94dbe07df2f5c`
(chunk `4d3182`). The resulting progress SHA is
`3244cc76143be0edf8f79871e891b044462e02d1afda68a2d23f0dc354f600ec`.
The compiler and full100 handoff remain live. Two complete memories and this
verified partial third memory do not satisfy the joint accuracy/latency target.

## Offset 020 third complete dependency wave

All 353 third-wave dependencies are resolved: 309 outputs passed immediately,
and all 44 oversized outputs passed their first bounded repair. The 100 batches
used 1,595.15 generation seconds at 24.24 aggregate tokens/s, with 5.180 GiB
peak allocation.

The compiler published 151 additional parents, reaching 165 complete source trees
and 405 parents, then began 314 further dependencies. Independent replay
authenticated every request/response and verified all 570 original leaves in the
completed trees unchanged, including raw-span descriptors. It made zero model
calls and read no raw transcript text.

The verification is `offset020-third-wave-verification.json`, SHA
`670bbcab03e48f0011245632b61834df87931ba5b3f1ca8f5179ab8f9db8ab3f`
(chunk `cd0724`). The resulting progress SHA is
`a9f46ee208c2baf5c72d82ebf8ed5345a9a52434f0d3031d7e2b524cef7b094d`.
Both compilation and the scheduled full100 handoff remain live. Two memories
are complete; the third memory and joint accuracy/latency result remain pending.

## Further batching assessment

A read-only check considered whether eight independent sequences could shorten
the remaining build. No eight-row measurement exists in the local compiler logs.
At the check, `nvidia-smi` reported 8,192 MiB total, 7,551 MiB used and 439 MiB
free (chunk `c4f9ce`). The compiler's roughly 5.2 GiB peak tensor allocation is
not a measurement of total available device memory. Reusable allocator memory
was not measured, so this observation neither proves nor rules out an eight-row
batch fitting in the same process.

The existing handoff explicitly binds the four-row compiler command, adapter
hash, measured batch result and common parent-method policy. A batch change would
need a measured successor and a corresponding handoff while retaining the
completed-output snapshot. No larger-batch probe or compiler interruption was
performed; the current four-row build and full100 handoff continue unchanged.

## Offset 020 fourth complete dependency wave

All 314 fourth-wave dependencies are resolved: 281 outputs passed immediately,
and all 33 oversized outputs passed their first bounded repair. The 88 batches
used 1,399.03 generation seconds at 24.34 aggregate tokens/s, with 5.171 GiB
peak allocation.

The compiler published 270 additional parents, reaching 219 complete source trees
and 675 parents, then began 260 further dependencies. Independent replay
authenticated every request/response and verified all 894 original leaves in the
completed trees unchanged, including raw-span descriptors. It made zero model
calls and read no raw transcript text.

The verification is `offset020-fourth-wave-verification.json`, SHA
`edf9b6074217a71419abcff5e931dea65b372e4f95fb0711afd23edf0b983fac`
(chunk `142dcc`). The resulting progress SHA is
`b1cae1620677a44ba3c770e16342518ef72841d9dd8444ab65567658d39bb82d`.
Both compilation and the scheduled full100 handoff remain live. Two memories
are complete; the third memory and joint accuracy/latency result remain pending.

## Offset 020 fifth complete dependency wave

All 260 fifth-wave dependencies are resolved: 239 outputs passed immediately,
and all 21 oversized outputs passed their first bounded repair. The 71 batches
used 1,019.13 generation seconds at 28.18 aggregate tokens/s, with 5.195 GiB
peak allocation.

The compiler published 353 additional parents, reaching 282 complete source trees
and 1,028 parents, then began 197 further dependencies. Independent replay
authenticated every request/response and verified all 1,310 original leaves in
the completed trees unchanged, including raw-span descriptors. It made zero
model calls and read no raw transcript text.

The verification is `offset020-fifth-wave-verification.json`, SHA
`ea22df970bf2f4c5798b2fd696f49e24ca5164882fd594a51739962af5982b12`
(chunk `ea3d95`). The resulting progress SHA is
`df2e114cae82ef308106a5f0984d96e14b3fb1ecaa536d67b625f3866ef5ab48`.
Both compilation and the scheduled full100 handoff remain live. Two memories
are complete; the third memory and joint accuracy/latency result remain pending.

## Offset 020 sixth complete dependency wave

All 197 sixth-wave dependencies are resolved: 179 outputs passed immediately,
and all 18 oversized outputs passed their first bounded repair. The 55 batches
used 795.00 generation seconds at 27.80 aggregate tokens/s, with 5.169 GiB
peak allocation.

The compiler published 471 additional parents, reaching 362 complete source trees
and 1,499 parents, then began 117 further dependencies. Independent replay
authenticated every request/response and verified all 1,861 original leaves in
the completed trees unchanged, including raw-span descriptors. It made zero
model calls and read no raw transcript text.

The verification is `offset020-sixth-wave-verification.json`, SHA
`e8b01d58a888ae9e3028ca0cb1f9cfb4a4391e6b054b118576c18619b3b6684d`
(chunk `38bffa`). The resulting progress SHA is
`6f81ddb48c70818671781e7849216b9eb09dc55504cbed29e15be1e194cc388e`.
Both compilation and the scheduled full100 handoff remain live. Two memories
are complete; the third memory and joint accuracy/latency result remain pending.

## Offset 020 complete and independently replayed

The third memory completed all 479 source trees, 2,214 parents and 2,693
unchanged original leaves after 2,385 new generation jobs across 605 batches
(compiler chunk `3eeb8c`). No dependencies remain. Its complete raw turns contain
1,045,527 `cl100k_base` tokens, as independently counted earlier.

| Artifact | SHA-256 |
| --- | --- |
| Complete hierarchy | `e8380ecdfd89bfba36f3f76d1ea84cdf7103c5370b4899b376480c4166e05650` |
| Final progress | `587147c13a1f66f933cb14e04a099320fb5b56fcf9402892c7c9817aadc21209` |
| Topology | `9e41c011204edb4cb1adbad279b841bff0596e471ada0ff8f921eba395166bce` |
| `third-namespace-verification.json` | `2503fe6a15083a4cec0a408d59afb04a5c27a1a53e946268152c3ea62016c8d0` |

Independent replay reconstructed the entire namespace with generation forbidden
and a zero-job allowance. It produced the identical hierarchy, preserved every
original leaf and raw-span descriptor, and confirmed the same method policy as
offset 000. The model was never loaded; new local and remote calls were both
zero. Replay session 16945 exited successfully (chunk `9d9d66`). This whole
namespace verification also covers the dependencies completed after wave six.

Three of ten memories are complete. The original compiler started offset 030
from 30 complete source trees and 17 parents, with 443 initial dependencies.
Its first four batches are saved (chunk `f9fb58`); handoff session 61096 remains
live and waits for all ten memories (chunk `936190`). These checks establish
reconstruction integrity; the full100 answer-accuracy and query-latency result
is still pending.

### Gateway recheck during offset 030 compilation

One synthetic Qwen readiness request at 2026-09-11 10:34 UTC again returned
HTTP 500 with an unknown-model error naming `qwen3-8b-gguf`. The request had a
30-second timeout, a 64-token output limit and zero automatic retries. No raw
corpus or benchmark questions were sent, and the local compiler was not
interrupted. Its next observed batch brought offset 030 to 224 saved initial
attempts (chunk `21e66a`); the full100 handoff remained live.

The sealed observation is
`eval_results/qwen-gateway-readiness-20260911T1035Z/observation.json`, SHA
`30fe2ff58da908eac079388f4baa5a4ef31b24cb72dee444c2261b91a01bc6ea`
(readiness session 56759, exit 0, chunk `d815ce`). Its preflight SHA is
`c2182ad7e8cb5715cf5c7e5f1ec8d37e02dd9ad003a15cf9bbffa0b048861342`.
Continue the local fallback and its existing handoff.

## Offset 030 first complete dependency wave

All 443 initial dependencies are resolved: 406 outputs passed immediately,
and all 37 oversized outputs passed their first bounded repair. The 121 batches
used 1,884.85 generation seconds at 24.55 aggregate tokens/s, with 5.188 GiB
peak allocation.

The compiler published 85 additional parents, reaching 70 complete source trees
and 102 parents, then began 403 further dependencies. Independent replay
authenticated every request/response and verified all 172 original leaves in
the completed trees unchanged, including raw-span descriptors. It made zero
model calls and read no raw transcript text.

The verification is `offset030-first-wave-verification.json`, SHA
`23d21eceaf5b42a00daa75bb2978b54c85de2fb94eefd387d92171aa3aed6344`
(chunk `8359d1`). The resulting progress SHA is
`aecfc78e788a781a007af00ff94d12ff7efd988122af3c35f34fea7b338efc89`;
offset 030's preflight SHA is
`3e5520e7245a0b2cc329bc083047ca99f2680a4b3c9a99611e903ff198a8c27d`.
Compiler session 65191 and handoff session 61096 remained live at publication
(chunks `6ccfd2` and `b56cae`). Three complete memories and this verified partial
fourth memory do not establish the full100 joint accuracy/latency target.

## Offset 030 second complete dependency wave

All 403 second-wave dependencies are resolved: 373 outputs passed immediately,
29 exceeded their summary budgets and one was not valid JSON. All 30 rejected
outputs passed their first bounded repair. The 109 batches used 1,682.45
generation seconds at 24.65 aggregate tokens/s, with 5.171 GiB peak allocation.

The compiler published 79 additional parents, reaching 98 complete source trees
and 181 parents, then began 375 further dependencies. Independent replay
authenticated every request/response and verified all 279 original leaves in
the completed trees unchanged, including raw-span descriptors. It made zero
model calls and read no raw transcript text.

The verification is `offset030-second-wave-verification.json`, SHA
`511ad42d03b34e57cd011f3fa4b7eedebce70f4e565fed6fd2bca4d9e6af34d2`
(chunk `f3ae39`). The resulting progress SHA is
`3ebae48a8da0fc730bd4fb98a17db920ef657da2b82b68b3ebe6be8fa04bd42f`.
Compiler session 65191 and handoff session 61096 remained live at publication
(chunks `26b409` and `fdb8c3`). Three memories are complete; the fourth memory
and the full100 joint accuracy/latency result remain pending.

## Offset 030 third complete dependency wave

All 375 third-wave dependencies are resolved: 342 outputs passed immediately,
and all 33 oversized outputs passed their first bounded repair. The 103 batches
used 1,565.60 generation seconds at 25.13 aggregate tokens/s, with 5.171 GiB
peak allocation.

The compiler published 246 additional parents, reaching 157 complete source trees
and 427 parents, then began 316 further dependencies. Independent replay
authenticated every request/response and verified all 584 original leaves in
the completed trees unchanged, including raw-span descriptors. It made zero
model calls and read no raw transcript text.

The verification is `offset030-third-wave-verification.json`, SHA
`1f3ec802f3fe88d4d325cb05f833dba436a1a20a40836bb6b35e0f68dd3601f7`
(chunk `a7a050`). The resulting progress SHA is
`a3ce269ef9173f5e0c7f17f6fe65c68426cf0232c4e78bee0e2ea9472c283cbe`.
Compiler session 65191 and handoff session 61096 remained live at publication
(chunks `548636` and `dcc4ad`). Three memories are complete; the fourth memory
and the full100 joint accuracy/latency result remain pending.

To verify the latest recorded wave artifact from this worktree:

```powershell
Get-FileHash -Algorithm SHA256 -LiteralPath 'eval_results/full1m-spine-parents-local-batch4-20260910-r1/offset030-third-wave-verification.json'
```

The hash must match the receipt above.

## Terminal JSON failure and successor

The compiler later reached 345 complete source trees and 1,483 parents in
offset 030, then stopped after one response exhausted both semantic repairs.
Session 65191 exited 1 (chunk `5e389e`), and handoff session 61096 exited 1
without evaluation calls (chunk `a04624`). The saved final response contains an
invalid apostrophe escape; removing only that encoding error makes the existing
parser accept its 49-token summary within the unchanged 128-token limit.

The original artifacts and implementation remain intact. Continue with the
JSON-recovery compiler and handoff documented in
[Research Log 180](180%20-%202026-09-11%20-%20Local%20Qwen%20JSON%20encoding%20recovery.md).
The successor preserves the three completed memories and resumes offset 030
from 413 complete source trees and 1,890 parents, with zero preparation model
calls. Do not restart the terminated original commands.
