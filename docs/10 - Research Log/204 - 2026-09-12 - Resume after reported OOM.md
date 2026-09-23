# Resume after reported OOM

**Status:** Resumed at the user's instruction. Two additional source requests
have verified recovery outputs, including one repaired section. The R3
continuation is producing accepted responses with concurrency two. Its repair
coordinator and full-corpus pipeline are live. The native joint full100
benchmark remains pending.

## Observed state after standby

The original local Qwen parent process survived: PID 54716, creation time
`1789232865.1039636`. Its third 128-job invocation completed 13,156 of 13,468
body hierarchies. The report SHA is
`c05b28a2fc05ddafbadaecab7c040faceb08b0853a6be5c58cace78ef5de73d8`.
It has entered the next invocation. The vector and reader-check processes also
survived with their original identities and continue waiting on their inputs.

The R2 ingestion continuation stopped after 79 responses, ordinals 8099–8177.
Requests 8178 and 8179 returned HTTP 500; 5,632 requests, ordinals 8180–13811,
were never dispatched. The terminal SHA is
`583c48dec549f1130e9fc28927ea65f7f08997c8f7f2fb5a51d263777c987472`.
PID 66728 is gone and session 55088 exited one. These observations establish
the execution state; they do not identify the cause of the reported OOM.

The R2 repair coordinator and full-corpus pipeline also exited one at their
dependency checks. The coordinator completed zero new cohorts and used zero
new calls. No full-corpus stage was released. Their failure SHAs are
`5619ff94b5e6509b1e9028c28bc0843003ed50ce6903089beb65668d65383e6b`
and `953f36fbcb93d01b413e12ae6e1f005394a82ac9a2e00d772f626596f9131ebb`.

## Recovery and unchanged evidence

`eval_results/native-spine-transport-recovery-20260912-r3` records two explicit
sequential replacement calls, with zero automatic retries. The existing
recovery reader used the actual R2 terminal report directly and authenticated
the unanswered original journals before and after execution. Original provider
execution remains uncertain; these are explicitly additional calls.

Request 8179 passed validation. Request 8178 retained 19 valid summaries and
needed one rejected section replaced with three exact subdivisions. That repair
completed in one call under
`eval_results/native-spine-recovery-section-repair-20260912-r2`. The resulting
section list covers all original raw content and preserves the 19 valid summaries.

| Artifact | SHA-256 |
| --- | --- |
| Transport R3 preflight | `a1a9d14e284f516d24b75adc85617ccecf7ef43cbbff9fa7abd3ce98cfe1952b` |
| Transport R3 policy | `d8f24e2760bb60cbe33d2cfbcf4f60a6c7af9281038ca4dd7d02e93c09fbf4bc` |
| Transport R3 result | `7048992acf1d1c981e54bc16e370d70fe22177554ab2a2f0f284a827b5ac5dab` |
| Section repair preflight | `e6f91577a4a4839b763a8e5ae3546ba0e0199e61d023aa0bab1abd14b33a5074` |
| Section repair result | `75125d2961530bd700da32cb7f24d64dfccb721870f6fc4d58d2418ede817a4e` |
| Section repair completion | `c1a5827daf87bd1e7a5ab1391182922704076d98ed51ce73f53415ff7dc3c4c1` |

The transport result intentionally still records 8178 as invalid; admission
must include its separate section-repair root. The two completed recovery
processes exited zero in sessions 50741 and 41151. Replays performed by the
section repair made zero new transport calls and reproduced the same result SHA.

## Successor ownership

Driver `.tmp/resume_unsent_native_batches_20260912_r3.py` authenticates the
original terminal report and both retired continuations. It excludes every
completed original and all 16 uncertain originals. It also checks that none
of its 5,632 selected requests has a checkpoint or validation before dispatch.
Preparation exited zero in session 32551. Policy SHA:
`b7e6e7b2b297138532690a03883576a65d897e95ce6183e12f2b218410f45974`.

The control root is
`eval_results/native-spine-complete-body-summaries-20260912-r1/continuations/unsent-20260912-r3`.
Execution session 69208 revalidated all inputs and accepted its first request,
ordinal 8180, before dispatching at concurrency two. The first-response SHA is
`84d806f8e67bdb6699e1c4bde652e513c3043534a3fd1553fba358c697beb5d7`.
It has now accepted all 14 requests through ordinal 8193, with zero automatic
retries. No additional local model was loaded.

Active successor configurations are
`.tmp/native-spine-source-completion-20260912-r3.json` and
`.tmp/native-spine-full-corpus-pipeline-20260912-r3.json`.
They retain all 372 completed original repairs, all 16 transport recoveries,
both recovery-section repairs, and the existing exchange/attention/parent/vector
work. The remaining source-repair allowance is 1,994 calls after the new
one-call section repair; R2 consumed none of its 1,995-call allowance.

Source preparation exited zero in session 65841 after replaying all 372
repaired originals and 16 recoveries, including both recovery-section repairs,
with zero new calls. Its first live scan recorded 7,792 accepted and 381 invalid
original validations; nine rejected originals were still awaiting new repairs.
Original invalid validations remain preserved after successful repairs.

| Process | PID / creation time | Session | Policy SHA |
| --- | --- | --- | --- |
| R3 ingestion | 64768 / `1789247682.877373` | 69208 | `b7e6e7b2b297138532690a03883576a65d897e95ce6183e12f2b218410f45974` |
| R3 source completion | 22440 / `1789248081.729816` | 37394 | `5a0559e56414d466d4fa9e8027be7b6bf24399825364d7895f9ff27e170d950f` |
| R3 full-corpus pipeline | 61000 / `1789248120.2750444` | 3846 | `9588154e7c4582eb21ea19049696580e4f20bddfe9b86828d725c372b9dbde33` |

Their start receipts are respectively
`640e30eb9790724b706f755f21cdb5fbff40440e945bde2a87bf5d359f935971`,
`4b9673bb1de9cb15f6f6a8b592cac3b05f2c18bbcb295d54fe44be98ebde43a0`,
and `22c2651c36ac68fb8c288f7acf7563853697562beee64e36eccb5a7c402eedba`.
All three processes were verified live with matching creation times and no
terminal records. The existing parent, vector and reader-check processes were
also verified live. Parent generation had reached job 96 in its fourth 128-job
invocation, with 1,754 accepted merge keys including inherited keys.

The R3 full-corpus pipeline waits for ingestion 64768, source completion 22440,
parent compilation 54716 and vectors 17192. The completed exchange and attention
dependencies remain unchanged. No full-corpus stage or benchmark answer has
been released by this successor yet.

All 115 implementation files bound by the prior full-corpus pipeline remain
unchanged. This handoff changes execution ownership and offline concurrency.
It makes no new accuracy or latency claim: completion still requires the same
fresh full100 population, at least 95% accuracy, and all eight matched latency
ratios at or below 1.10 in the same run.
