# Gateway timeout recovery and ingestion continuation

**Status:** All eight newly timed-out source requests have recovered successfully.
The replacement original-request process has resumed dispatch with concurrency
four. Its replacement source-completion coordinator and full-corpus pipeline are
live, retaining completed repair work. Local Qwen parent compilation continues.
No native joint full100 result exists.

## Retired ingestion and preserved work

The original-request continuation stopped after 5,350 completed batches and
eight `APITimeoutError` failures at ordinals 8091 through 8098. It drained its
in-flight requests and sent no subsequent batch. Exactly 5,713 prepared original
requests, ordinals 8099 through 13811, remain never dispatched. The old process,
PID 56400, is gone and session 98453 exited one.

Its source-completion coordinator and full-corpus pipeline also exited one at
their dependency checks. The coordinator had completed six repair cohorts using
53 calls, and no repair cohort was left in flight. The pipeline had released no
full-corpus stage or benchmark answer call. Their failure records remain intact.

| Terminal evidence | SHA-256 |
| --- | --- |
| Original continuation `finished.json` | `6900d1319d4d359b9bc491d5ab56bf9ab4e59d7fb93dfe54f0c6778f14df4b77` |
| Source completion R1 `failure.json` | `671c9d8ba156f82f40b969f3928de535bf8a423ce4d799e98345ad1d61afd246` |
| Full-corpus pipeline R1 `failure.json` | `c7741b2570867e30dc894d4b313eea269367ade3bd1cee7adca04ce4df604976` |

The last original validation scan has 7,709 accepted and 376 invalid batches.
There are 372 repaired originals, with four additional rejected batches still
awaiting repair. The earlier six transport recoveries remain separate.

The sixth automatic cohort, 0005, repaired 13 original batches in nine calls,
preserving 272 valid summaries. Its first stage accepted 32 of 42 replacement
sections; the second stage retained those 32 and replaced the ten rejected
sections with 23 accepted subdivisions. Final root:
`native-spine-source-completion-20260912-r1/cohorts/0005/direct_roots/stage-01`.
Its result SHA is
`41ef1e8a6376cbb5ec49f4ebde1d62836cce10a6327713f517a826e3a798713f`;
cohort completion SHA is
`3002ea7c9001461efecc9d632a67bf558969297035d0fd5e4558f60014e7a97e`.

## Eight explicit transport recoveries

`eval_results/native-spine-transport-recovery-20260912-r2` contains eight separate
replacement requests. The recovery ran sequentially, with zero automatic retries,
through the already-authorized Terra gateway. All eight responses were accepted,
and the existing verifier rechecked every unchanged original request journal
before and after execution. Session 70009 exited zero.

The old continuation terminal schema omits fields required by the existing
transport recovery reader. The new driver therefore publishes an explicitly
labelled derived compatibility report. It binds the actual original policy,
start record, terminal record, retired process and exact timeout ordinals; it
does not replace or claim to be the original runner report. The recovery reader
independently reconstructs each unanswered original journal from its exact input.

| Recovery artifact | SHA-256 |
| --- | --- |
| `source-continuation-failure.json` | `5fc404f5f931951e305a84a323d3ad30713551478f6109b428dac7a601678f85` |
| `preflight.json` | `8d4b003d617a40e1325acf5273a0840bce706a7b466dcd6be69ed192f4fab8c2` |
| `execution-policy.json` | `da7419c63020a7f13c13798f16dcb1c006ce36a7a43fe8eb916f6c40776eaaa4` |
| `result.json` | `838b017fd5c3ce4b4ef00e7ffd7783ed6d2ea23c0d7e307a40f3ae3c636696cd` |
| `finished.json` | `77db3e8fb766340714c8aff8a6de5b05b4bbb2aca91021c241c4963510ca6615` |

Driver: `.tmp/recover_native_transport_20260912_r2.py`, SHA
`5ed87c5206bc86573196d60f9f9c1bb30c6de95c413784e3f0a17023e6d72f89`.
Recovery process 55864 / creation `1789242439.7991848` has completed. These eight
requests are added to the earlier six transport recoveries, for fourteen total.
Original provider execution remains unknown for the timed-out calls; the eight
replacement calls are explicitly additional calls.

## Replacement continuation

The replacement scope authenticates the original terminal population and the
retired continuation, verifies all 13,812 prepared requests, and requires no
checkpoint or validation to exist for any of its 5,713 selected ordinals. It
excludes all fourteen uncertain originals and all previously completed requests.

| Item | Value |
| --- | --- |
| Control root | `native-spine-complete-body-summaries-20260912-r1/continuations/unsent-20260912-r2` under `eval_results` |
| Policy SHA | `fbcab4ee4ef7e671d2307fe3b7677f74f996a4ff2c1610449a7d8eca8b17d577` |
| Driver | `.tmp/resume_unsent_native_batches_20260912_r2.py` |
| New original ordinals | 8099 through 13811 |
| Maximum new initial calls | 5,713 |
| Dispatch | One fresh request first; concurrency four only after a response |
| Automatic retries | 0 |
| Session | 55088 |
| Process / creation time | 66728 / `1789243039.4647057` |

Read-only preparation completed successfully in session 78951. The execution
process revalidated the same inputs, published its start receipt and accepted
all 24 summaries from its first new request, ordinal 8099. It then continued
dispatch with concurrency four. Its start SHA is
`85c173e93af98e882b948b61d29963338bead4b262463276310b5c5163b7a2fb`;
first-response SHA is
`2ec1452e58e21224474231b808675dac7acb46a3d89edc8824ae519cda4e2f57`.
Concurrency was reduced from eight to
four after the timeout burst; this changes offline dispatch, not the matched
benchmark's serial timing policy or Qwen input policy.

The replacement coordinator configuration carries forward all six completed
cohorts and all 372 repaired originals. It includes both transport recovery roots
and the existing repair of the earlier invalid recovery. Its remaining repair
allowance is 1,995 calls, preserving the original 2,048-call allowance after the
53 completed calls. It will assemble a complete store only after every original
request has an accepted or explicitly repaired/recovered disposition.

Active configurations:

- `.tmp/native-spine-source-completion-20260912-r2.json`
- `.tmp/native-spine-full-corpus-pipeline-20260912-r2.json`

Their roots are `native-spine-source-completion-20260912-r2` and
`native-spine-full-corpus-pipeline-20260912-r2`, under `eval_results`. The full
pipeline consumes the new coordinator's `complete-body-store`, reuses the same
exchange, attention, parent and vector populations, and retains the unchanged
full100 accuracy and latency gates. Both successors are now running. Source
preparation replayed and authenticated all 372 repaired originals and fourteen
transport recoveries without new provider calls, then exited zero in session
32114. The complete-corpus pipeline remains unreleased while its four live
dependencies finish: new ingestion, new source completion, existing parent
compilation and existing vectors.

| Active successor | Source completion R2 | Full-corpus pipeline R2 |
| --- | --- | --- |
| Policy SHA | `c3a8fabc8e4bfc5e736f8d59d8af41598272bf611c72d8a33b3d893d25201415` | `48676fa80057bcfb7a2c1805f9839f4cef127a7a42a19bdb381a1ffae9e0cd88` |
| Started SHA | `140ec097073341a0ddcc8350d3a8dd4f20373dc3bb42b5bf9c666288f1780dec` | `ad442f53e4df929616214ffe302a431d63b5b87555ce7a9aa0d82e93a3b739bc` |
| PID | 64516 | 532 |
| Creation time | `1789243454.2342727` | `1789243491.013523` |
| Session | 30099 | 28679 |

All three replacement processes were verified live with their matching creation
times and no failure records. All 115 implementation files bound by the new
full-corpus pipeline remain unchanged. Its full-corpus stages have not started.
The latest scan has 7,742 accepted and 377 invalid original validations, with
five rejected batches awaiting automatic repair. The replacement coordinator
has made zero new calls so far; its retained predecessors used 53 calls.

## Local parent compilation

The local parent process remains live with its original identity. Its first
128-job invocation accepted 120 new summaries and completed 12,914 of 13,468
body hierarchies. The partial report SHA is
`0d7066eeaa9cf4cbd1a15206be386574f0654730519674301afc4cbf892bf0e2`.
There are 554 bodies with pending merges. The next bounded invocation is running.
An earlier audit of 76 generated responses found 71 accepted summaries and five
generations without a stop; all stopped responses passed parsing. The existing
48-word and 24-word recovery prompts remain unchanged.

The completed attention windows, exact raw addresses, accepted source summaries
and completed reader construction check remain available. This recovery changes
execution ownership and explicitly accounts for additional transport attempts;
it establishes no new accuracy or API latency result.
