# Ingestion complete and final corpus repair

**Status:** All original ingestion requests now have completed or explicitly
recovered dispositions. The R6 parent hierarchies, vectors and real reader check
completed successfully. The repair coordinator reached its 24-hour waiting
limit before ingestion ended; its successor has repaired the final 11 rejected
batches and is assembling the full source store. The full-corpus pipeline is running and waiting solely for source
completion. No native joint full100 accuracy or latency result exists yet.

## Completed ingestion

The R3 continuation completed all 5,632 selected requests, with zero transport
failures and zero requests left undispatched. Its original 16 uncertain request
journals remain excluded; their separately verified recoveries and section
repairs remain required inputs to admission. The continuation process is gone.
Its `finished.json` SHA is
`35e7f8085b07ca2421d3171ae36b8f6f03f5a6633ca0f782bb65b1bada710a34`.

Completion of dispatch is distinct from complete source admission: rejected
summary responses still require the exact-subdivision repair path.

The final strict validation scan accounts for all 13,812 original requests:
13,172 accepted validations, 624 rejected validations and 16 explicit transport
recoveries. Of the rejected originals, 613 have retained repairs. The final
11 rejected ordinals are 13620, 13657, 13684, 13698, 13703, 13723, 13726, 13727,
13739, 13797 and 13808. The scan exited zero in session 92091 and confirmed no
overlap between completed originals and explicit transport recoveries.

## Completed local hierarchy and serving checks

The surviving parent process finished all 13,468 bodies in the R6 partial
source store. It used 713 new local generation jobs across 179 batches and six
invocations. Its final result contains 64,180 leaves, 50,712 parents and 140,256
atomic summaries, preserving the original raw addresses. No raw content went
to Qwen. This completes the available R6 hierarchy, not the full 31,166-body
corpus hierarchy.

All 139,694 prepared summary vectors also completed. The queued real reader
check then instantiated all 100 partial namespaces, bound all 13,468 templates
and exactly hydrated 3,818 spans across both retrieval arms. It preserved all
baseline sections and added 52 sections. The histories in this check contained
420,079–567,413 body tokens and all rejected full-1M admission. It read no
benchmark questions or gold answers and measured neither answer accuracy nor
matched API latency.

| Completed artifact | SHA-256 |
| --- | --- |
| Parent result | `5db97f57098faeb4cc69663eeb723f5435f75dd6c29d4f6eb3c445f2972f1e98` |
| Parent completion | `8a3e358a52781904176fcba83907c41d5457a176955902584804be7e1cb6be31` |
| Vector result | `7ad2ef941617ab0a2b7c7529c9ccb162cf2498299d7d82890379901b9324d7ab` |
| Vector completion | `3632a199320ab493a4f65eaff3fc1327ea98437d6df917f8e0fbc581bd39a914` |
| Reader result | `c0e7a0de3077371f9c33d9de7db46c038c25a41c153f6a03e94310cfffe0fd26` |
| Reader completion | `4e789e8896bd37346483081d6c51a77c7a5e1934570a5c7906ecab32c90c45fd` |

All corresponding original process identities are gone. These completed
artifacts are reused by the successor full-corpus pipeline.

## Coordinator timeout and retained repairs

The R3 source coordinator completed 88 cohorts, repairing 241 additional
original requests in 197 calls. Together with its 372 initial repaired
originals, these account for 613 repaired originals. Every cohort has a sealed
completion record; no incomplete cohort or final-store assembly was left
in flight. The last cohort, 0087, has completion SHA
`797f3437d95c29e9a20a1cbd39fae4aa05bc053567f0a08c0730d9c0e401bd1d`.

The actual traceback identifies the coordinator's 86,400-second time allowance,
not a transport timeout. Failure SHA:
`9a54cf8da4cbfbfbf982bfef17b19bf4cf59f8c9f93d16dde252a6bc96e58e1c`.
Its session 37394 exited one. The dependent full-corpus pipeline also exited
one, reporting the missing successful source-completion artifact. Its failure
SHA is `df86749a98441a8f704bda4a270ffb110d94b5bf0089f5cf0c0392cad6d1ccee`.
It had released no full-corpus stage or benchmark call.

The successor configurations are
`.tmp/native-spine-source-completion-20260913-r1.json` and
`.tmp/native-spine-full-corpus-pipeline-20260913-r1.json`.
They retain every completed cohort's final repair root and all prior transport
recovery roots. The remaining repair allowance is 1,797 calls, subtracting the
197 completed calls from the prior 1,994 allowance. The source producer remains
the successfully completed R3 continuation, so there are no original requests
to restart and no producer waiting period to repeat.

The recovery receipt is `.tmp/native-spine-resume-after-timeout-20260913-r1.json`,
SHA `3e944d9923a0d9bd36b4d448456ad91ff38c17f34a7626b44fc6422eba74ba40`.
All 115 implementation files bound by the previous full-corpus pipeline still
match. The same complete-source, exact-hydration, full100 accuracy and matched
latency gates remain required.

## Active successors

Preparation completed in session 67685 with zero new model calls. It replayed
and authenticated all 613 repaired originals and all 16 transport recoveries.
The new source coordinator immediately identified the 11 remaining rejected
batches after checking the successful terminal producer record. Its first
repair stage preserves 246 valid summaries and requests 31 replacement sections
in four calls. Preflight SHA:
`0b42231bbe237d56cb89fef644ee2d383f8b9f89e088d85573bc07c52baaa200`.
Further bounded subdivision is permitted only for sections that still fail
validation; the old source requests and valid summaries remain unchanged.

| Successor | Source completion | Full-corpus pipeline |
| --- | --- | --- |
| Root under `eval_results` | `native-spine-source-completion-20260913-r1` | `native-spine-full-corpus-pipeline-20260913-r1` |
| Policy SHA | `053efd7699667a2711b45d8df5ca56ddb763c759fb01c667d0b23bcb60cda305` | `a5e3299cad569c1fe5686760ef2423e318fdb4aedc250af832541da42c0106a5` |
| Started SHA | `8ab08a2ad0a0a978f588d02d66bc44592fbe7d7f6f21e9232b68014ec4289cbc` | `7efacd7861450eb2cfa0f9106807df1aa4e4735c81b472c3f096cae14fb03e95` |
| PID | 35852 | 54408 |
| Creation time | `1789364598.0538237` | `1789364624.9737024` |
| Session | 88985 | 14093 |

Both processes were verified live with matching creation times and no terminal
records. All other full-corpus prerequisites are complete; the pipeline now
waits solely on source completion 35852. Its full-corpus stages have not yet
been released. No new original ingestion request is scheduled.

The final repair cohort is now complete: all 31 replacement sections passed
in four calls, covering all 11 pending original batches. Its result SHA is
`3b4979c1da448b1f7eadff04209e04e4bb50b8c9a11350d7df2a62f39df84e32`;
cohort completion SHA is
`644e301894ee5c1d5228577cba143140dda2b6a59bcb3c065bf94ebe501353e7`.
The subsequent strict scan found zero remaining rejected batches needing
repair. All 624 rejected originals now have retained repair results, alongside
13,172 accepted originals and 16 transport recoveries. The source coordinator
has entered final store assembly; successful 31,166-body admission and
preservation of the R6 store must still complete before releasing the pipeline.
