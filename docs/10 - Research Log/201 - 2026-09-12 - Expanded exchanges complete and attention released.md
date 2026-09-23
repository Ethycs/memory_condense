# Expanded exchanges complete and attention released

**Status:** Exchange compilation completed all 13,468 bodies in the R6 snapshot.
Attention preparation completed and its execution child is running. Full source ingestion
and the native joint accuracy/latency evaluation remain incomplete.

Recent continuations were verified waits on the existing live workers. They
observed accepted source responses, completed repair cohorts and new Qwen merge
results without starting duplicate work. This continuation observed exchange
completion and verified the next process transition.

## Completed expanded exchanges

Root: `eval_results/native-spine-exchanges-20260912-r4`.

| Artifact or measurement | Value |
| --- | --- |
| Input bodies / atomic sections | 13,468 / 140,256 |
| Completed exchange sets / exchanges | 13,468 / 70,218 |
| Result SHA | `e65ac10d17d96e150425e9708824e37719a10bf90dd49979e879e22868719cea` |
| Preflight SHA | `69f9162c6509042e798feb3750484634da1f7ef6e0703dd3a87d70d6e9471083` |
| Raw-span population SHA | `32129dcbed4dcd19cff479edabf995167f3fb7cff278209389615edd3637a7d5` |
| Continuation finished SHA | `ee70e345551669bd696ae2b37af9d3b2476320c5405da3ae3fd724d8f3b7e557` |
| New local jobs / batches / invocations | 437 / 121 / 4 |
| Remote provider calls / raw inputs to Qwen | 0 / false |

The continuation is under `continuations/2048-20260912-r1`. Session 55723 exited
zero; PID 49268 is gone. The run did not exhaust its 2,048-job allowance or stop
by request. Its final runtime cache count was 636 accepted merge keys, including
the 317 inherited keys. Exact atomic span coverage is checked before each body
exchange set is published. The final two bodies required further sequential
summary merges, which completed through the existing bounded output repairs.

This is completion of the available R6 snapshot. The result explicitly retains
`complete_source_compilation=False` and `full100_target_passed=False`. It is now
an eligible completed cache for the queued full-corpus pipeline to reuse.

## Attention handoff

Root: `eval_results/native-spine-attention-20260912-r5`.
The controller remains PID 42700, creation time `1789232720.9850051`, session
11051. It verified that the exact exchange process had exited and that its
terminal receipt matched the existing policy.

- Released receipt SHA:
  `f162045d492d5783dd68f87102c65d8e99c327ec9e369f415d203f31b5e99f18`
- Preparation child: PID 36168, creation time `1789237642.3902855`.
- Command module: `tools.prepare_expanding_native_spine_attention prepare`.
- Existing attention cache: `native-spine-attention-cache-20260912-r1`.

The preparation child authenticated the completed exchange producer and prepared
13,643 unique summary windows under the unchanged attention method. It exited
zero, and the controller launched the separate execution child, PID 56848,
creation time `1789237899.0096807`, using the same module's `run` phase. The child
was confirmed live. A stale-PID observation of the preparation child reported
`NoSuchProcess`; the controller's actual child list and successful prepare-exit
receipt established the normal process transition without restarting anything.

- Attention preflight SHA:
  `ba173bcc7ee82a4858f8fbbbbced738ace3e563f2edfdf1c2285d36c0e57573d`
- Producer admission SHA:
  `d92fd1865d22afcb0236db56d8093a1ddc011437960fe13fbf18e3aa41f16da5`
- Preparation exit SHA:
  `6e8ab5d9f87e05b9ae5ddde82f4e5eb0aad4804e57cfa7dca083962e33724f82`

The final attention result remains pending. Raw inputs to Qwen, generation calls
during preparation and query/gold inputs remain false or zero.

The full-corpus pipeline, PID 26652/session 24901, now reports five live
predecessors after accepting exchange completion. It has not released its own
model stages or evaluation. Expanded parents and vectors retain their current
owners and wait for the attention/parent sequence.

## Source repairs

Automatic cohort 0001 repaired another 12 original batches in five calls while
preserving 270 valid original summaries. Its final root is
`native-spine-source-completion-20260912-r1/cohorts/0001/direct_roots/stage-00`.

- Result SHA: `4b0e21da4b1829f7865dc671b4525bed8f5131cb3e3ef85d229120392063a554`
- Cohort finished SHA: `e6fd4df0a6284c129c8aad711425962e68055ce71c3f6aaf66d848ca19da9a8f`

The first two automatic cohorts therefore completed 37 originals in 24 calls,
bringing completed original repairs to 323, with six transport recoveries still
accounted for separately. Cohort 0002 selected 15 newly rejected originals,
preserving 328 valid summaries and initially preparing 49 replacement sections
in seven calls. Its initial preflight SHA is
`ce1455f3fca5f7b51526669c605d0acbdcf4a4d7393962eb72d3de16a3c692bd`.
The initial seven calls admitted 13 batches and 47 sections. One refinement call
replaced only the two remaining rejected sections with five exact subdivisions.
All 15 batches are now complete in eight calls, preserving the 328 original valid
summaries and all 47 accepted first-stage sections. Final result SHA:
`3d446df19b44458daae0b5fd4e274cf5d04a128eec891ec3d7993eb055ef1a42` at
`cohorts/0002/direct_roots/stage-01/result.json`. Cohort finished SHA:
`2e59552d6ec05339acf9b76a4a26a14e856f323f2daec680951fc9afa010cdd2`.
The three automatic cohorts total 52 repaired originals in 32 calls; including
their seed lineages, 338 originals now have completed repairs.

Original ingestion remains live. Its latest scan after cohort 0002 completed
showed 7,072 accepted and 340 invalid validations, with two awaiting repair.
No new tests or implementation
changes were needed for these automatic transitions. The full 31,166-body store,
complete hierarchy/vector population, and same-run 95/100 plus matched latency
result remain required.
