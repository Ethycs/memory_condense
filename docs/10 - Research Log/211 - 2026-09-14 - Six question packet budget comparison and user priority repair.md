# Six-question packet budget comparison and user priority repair

The same cached history contains 1,098,417 actual eligible body tokens. No new
history, summary, attention window, parent tree or document vector was compiled
in this continuation. The 100-history pipeline remains stopped while the design
is unsettled. The 95% joint benchmark objective remains unverified.

## Frozen design questions

A read-only scan of the existing S dataset found only the original bulb question
with all annotated support bodies present in this cached history. No second
official benchmark question could be added without changing the source scope.
Discovery is saved under `native-spine-cached-question-discovery-20260914-r1`.

Five additional questions were authored from actual user statements at fixed,
spaced source ordinals before routing those questions: the rejected house-offer
amount; the difference between camera and photography-manual prices; the
geographic coverage of a field guide; the database, interface, reviewer count
and timing requirements of a room-service blog; and charity donations. Exact
support quotations and reference answers are isolated in an evaluation file.
The retrieval inputs contain questions and source identities, not gold text.

The resulting six-question set contains **one previously exposed official
question and five manually authored design probes**. It is not a substitute for
benchmark accuracy. All questions use the cached history's original question
day, so the same actual 1M-token population remains eligible.

Use `eval_results/native-spine-design-questions-20260914-r2/questions.json`, SHA
`5f36b04fdfed7fda4cfeba57f152dcd635138a8c127803692967fbb8c5f1420b`.

## Fixed-set comparisons

Each six-question comparison made 30 fresh Terra streams: two memory methods,
their two identical-prompt API controls, and a short-chat control per question.
Pair order alternated. Twelve Sol judgments followed after all answers were
saved. Model, reader instructions, dated queries and output caps stayed fixed.
Retrieval and prompt construction were inside memory-call timing. Resident
setup was excluded and recorded separately. There were no automatic retries.

| Context cap | Method | Design answers correct | Median total | Matched API median total | Short-chat median total |
| --- | --- | ---: | ---: | ---: | ---: |
| 3,072 | User-first, protected prefix 4 | 6/6 | 5.006 s | 4.149 s | 3.411 s |
| 3,072 | Parent context, protected prefix 4 | 5/6 | 4.228 s | 3.795 s | 3.411 s |
| 1,024 | User-first, protected prefix 4 | 5/6 | 4.483 s | 4.214 s | 3.385 s |
| 1,024 | Parent context, protected prefix 4 | 4/6 | 4.568 s | 3.985 s | 3.385 s |

Parent context abstained on the bulb question in both runs; its matched API
control answered correctly. The smaller packets also lost the field-guide
answer in both methods and both matched API controls. Thus smaller context
alone did not meet the joint objective. These small samples do not establish
stable latency distributions or a population accuracy rate.

Parent context did add original raw sections on the five new questions at the
larger budget. That functional expansion produced no accuracy advantage over
user-first ordering in this set. Keep it experimental.

Reports:

- `native-spine-six-question-evaluation-20260914-r1/report.json`, SHA
  `788a39469a70ad576906a832fac0fc21a810ccf0d83d4162008f1edd102d468e`.
- `native-spine-six-question-evaluation-1024-20260914-r1/report.json`, SHA
  `3146d15c518f7ac3d0c287c7819b58d1574cb4185dd54b78813121345c484c53`.

## Concrete failure and repair

The field-guide query's highest lexical match was a 561-token assistant reply
about park trails. The protected direct prefix put that reply ahead of user
facts. Under 1,024 tokens, the hydrator admitted the long reply and rejected the
69-token user statement explicitly naming North America. Both matching methods
had selected the correct atomic address; packet ordering lost it during exact
hydration. Increasing attention expansion could not recover its missing budget.

`NativeSpineContextRouter` now allows `protected_direct=0`. With zero protected
prefix, user-first ordering applies across the whole direct shortlist. It still
retains every original candidate address, and the existing hydrator enforces
the rendered-token and read limits without clipping raw sections. Qwen remains
summary-only during ingestion; there are no live Qwen or raw-scoring passes.

The configurable runner is `tools/evaluate_native_spine_design_questions.py`.
Exercise the repaired candidate with `--context-tokens 1024 --protected-direct 0`.
The defaults retain the prior comparison settings. An explicit two-to-eight
question subset supports focused development checks without silently expanding
the corpus. Every subset and parameter is recorded in its preflight.

A regression test puts an assistant reply at rank one and proves that zero
protected prefix preserves two user facts under a two-read budget. Twenty-three
focused tests pass, including default-packet compatibility, source/date scope,
raw identity rejection, parent context and prevention of accidental 100-question
design execution.

## Focused real-data verification

The repaired configuration was tested on the bulb and field-guide questions:
ten fresh streams and four judgments, with both methods **2/2 correct**. This is
a focused regression result, not a fresh six-question score. Do not combine the
old five correct answers with this repair and claim a new complete pass.

The field-guide user-first packet now contains 12 user sections, including the
explicit coverage statement, in 1,005 tokens. Its answer took 3.410 seconds
versus 3.544 seconds for its identical-prompt API control. Parent context took
3.506 seconds versus 4.244 seconds for its matched control. The bulb observations
were slower: 6.342 seconds for user-first and 5.324 seconds for parent context.
Individual timing differences remain noisy, and short chat remains faster.

Report: `native-spine-user-priority-repair-20260914-r1/report.json`, SHA
`2423586b386704b798c9ca9052549bf71192d38922a4c4c6f1feb2cb7f001924`.

The next design check must run the corrected configuration freshly across the
frozen set, then cover temporal changes and multiple-session evidence. The
native full100 accuracy and latency gates remain outstanding; do not resume
broad preparation while these design checks are unsettled.

## Verification and preservation

All three processes exited zero. A separate audit verified 70 answer journals
and 28 judgments, identical memory/control prompts, prediction and raw-span
hashes, token counts, normal stop events, accuracy totals and recomputed latency
statistics. It made no new provider calls and found no other Python jobs in the
worktree. Audit SHA:
`3e366a94f4e5ad79000ab9514880c7e863fc7183ac7a53298ce19af4f4e3a8fe`
under `native-spine-design-comparison-audit-20260914-r1`.

Prior code was copied and hash-checked before changes. The first run has its own
`implementation-preservation.json`; further snapshots are under
`native-spine-context-code-snapshots-20260914-r1`. The audit verifies either the
current implementation or its exact preserved snapshot. Existing ingestion
artifacts and earlier results remain intact.
