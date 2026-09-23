# Full100 as-of result and reader failure audit

**Date:** 2026-09-10  
**Status:** all answers, judgments and verification complete; joint target failed; failure analysis active  
**Predecessor:** [165 - Ten complete memories and live full100 evaluation](165%20-%202026-09-10%20-%20Ten%20complete%20memories%20and%20live%20full100%20evaluation.md)

## Complete result

The as-of cutoff scored **80/100**, compared with **74/100** for semantic seeds.
Both methods pass the provisional 1.10 latency limit against their
identical-evidence API controls. Both fail the 95/100 accuracy requirement and
the short-chat latency limit. The goal remains active; neither method is
promoted as satisfying it.

All ten resident memories contain approximately 1.04M token proxies. Every
memory request recomputed query embedding, summary routing and exact raw
hydration inside its end-to-end clock. Resident setup is recorded separately.
All 500 new streamed Terra responses sealed before any judging. Sol then
performed 200 logical judgments through 136 physical calls, with zero initial
replay hits. The final report replayed those judgments and all ten source
admission records without new provider calls.

| Arm | Correct | Median total | p95 total |
| --- | ---: | ---: | ---: |
| Semantic seeds | 74/100 | 6.647500 s | 10.456848 s |
| Its identical-evidence API control | Unjudged | 6.261486 s | 10.649447 s |
| As-of cutoff | 80/100 | 6.580489 s | 9.366184 s |
| Its identical-evidence API control | Unjudged | 6.038933 s | 9.733663 s |
| Short API chat | Unjudged | 5.082916 s | 6.353070 s |

The cutoff's median/p95 total ratios are **1.089677 / 0.962247** against its
matched API control and **1.294629 / 1.474277** against short API chat. Visible
TTFT gives the same gate outcomes. All 500 answers arrived in one visible
content event; this run does not demonstrate an incremental streaming gain.
Median/p95 local preparation was 0.353309 / 0.446750 seconds for the cutoff.
The prompt and answer workload still cost substantially more than short chat,
even though the retrieval overhead meets the matched-evidence allowance.

The same reader, raw budgets, summary-only Qwen boundary and exact hydration
contract were retained. Qwen attention partitions the summary-derived leaves;
the current query path uses BGE summary addresses. This experiment compares
the date policy on that shared hierarchy, not attention against an alternative
chunker. Parent-summary routing remains deferred.

## Paired outcomes and reader variation

| Outcome | Questions |
| --- | ---: |
| Both correct | 71 |
| Cutoff correct only | 9 |
| Semantic seeds correct only | 3 |
| Both incorrect | 17 |

Using zero-based evaluation ordinals, the nine rescues are
`[5, 14, 16, 30, 43, 72, 74, 81, 95]`; the three losses are `[50, 58, 94]`.
These are diagnostic labels, never serving inputs or routes.

Thirty-five questions have byte-identical memory prompts between methods.
Ten of those pairs have different literal predictions, and five have different
verdicts: the cutoff invocation gains ordinals 14, 16, 72 and 74 and loses 58.
Thus **three of the six net gains occur without any prompt change**. The 65
changed-prompt pairs account for the other net three. Do not describe the full
six-answer difference as a causal gain from date filtering.

The cutoff and its direct API control also produce different literal answers
on 29 questions despite identical prompts. Those control predictions remain
unjudged; literal differences include harmless wording changes and do not
establish 29 accuracy differences. The target score always uses the actual
timed memory predictions, without selecting a better repeat answer.

## Initial packet inspection

The twenty cutoff misses are:

```text
13, 27, 34, 36, 42, 49, 50, 52, 53, 54,
58, 60, 61, 67, 69, 75, 82, 86, 87, 94
```

The sealed inventory includes both predictions, both judgments, the paired API
predictions, exact prompt identities and all 500 timing observations. The
following cases were read directly from the frozen raw packets:

- **50, follower growth:** the packet contains Twitter increasing from 420 to
  540 and TikTok gaining 200 followers in three weeks. The cutoff answer and its
  API repeat both choose Twitter. The relevant numbers are present; additional
  routing alone cannot explain this error. Long ROI and Instagram tutorials
  also occupy the packet, suggesting a useful evidence-density diagnostic.
- **58, combined reading duration:** the identical memory prompts explicitly
  contain two-and-a-half weeks and three weeks. One call answers five-and-a-half
  weeks; another answers 33 days. The packet also contains January/February
  reading dates for the same title and a later audiobook completion. The reader
  must distinguish occurrences and media before reconciling durations. The
  existing broad preference for explicit boundaries is a hypothesis to test,
  not a proven cause or permission to force the benchmark answer.
- **60, study abroad:** the raw packet names the University of Melbourne and
  Australia. The answer names only the university, and the fixed judge rejects
  the missing country. This is an answer-completeness issue under the recorded
  rubric, rather than a missing institution in retrieval.
- **75, accommodation difference:** the raw text says Maui costs **over $300**
  and Tokyo **around $30** per night. Both calls answer **over $270**; the judge
  requires exact $270. This is a source/precision ambiguity to preserve in the
  audit. Do not strip a supported qualifier or alter the frozen judgment merely
  to increase the score.
- **94, baking-class date:** the selected March 21 user statement says the
  class was yesterday, while another excerpt describes an April 10 birthday
  cake. The cutoff answers 26 days and its API repeat 25; semantic seeds answers
  the accepted 21. The needed occurrence is the class associated with the
  birthday cake. A source-level event audit is needed before crediting the
  semantic-seed answer as faithfully grounded or attributing the loss to the
  cutoff alone.

There are 603 context-budget diagnostics in semantic-seed requests and 579 in
cutoff requests. These count rejected sections, not failed queries or missing
answers. The inspected follower packet's diagnostics are all `context_budget`.
Determine which rejected sections contain necessary evidence before changing
admission. Do not infer that every wrong answer is a retrieval failure.

The next implementation work must address supported evidence selection and
reader behavior together. Separate missing event coverage, cross-occurrence
confusion, over-terse answers and irrelevant packet filler. Any reader or
packet successor needs its own frozen comparison, including previously
correct questions; repairing only these twenty cannot satisfy the full100
objective. Keep the original judgments and evaluate any ambiguous cases
separately. No confirmation200 data was opened in this continuation.

## Artifacts and terminal ownership

Paths are under this worktree's `eval_results/`.

| Artifact | SHA-256 |
| --- | --- |
| `full1m-spine-as-of-full100-20260910-r1/answer-population.json` | `e03e3f5ee6bf97a6b4db9e35b07b8fb16482abf8407ac506b3d4114be9207200` |
| Same root, `joint-full100.json` | `9032233bc67b88387214124a63ffbdeb5aa7212988dbe1dce724a359a4945755` |
| Same root, `complete.json` | `bd7729f70cb57df557e5524e0fa18ed588c1854d22902d8552416850237901d1` |
| `full1m-spine-as-of-after-compilation-20260910-r1/complete.json` | `d2257335cd846acce27b6c7049e23b089c1765e13e14c19dc8d2630e10433443` |
| `full1m-spine-as-of-failure-inspection-20260910-r1/inspection.json` | `37cffbc993b366e47bcb9e86caa74447a40b978120580e8e774de2079a4da390` |
| Same diagnostic root, `inspection-canonical.json` | `402be2fd7b4101cce03dd1107d2d1ec27bceb763a3e39441339ecd7e4370a2c9` |

The PowerShell inspection authenticates 500 request/response chains, all
answer/judge bindings and the matched prompts without model work. Its original
pretty-printed JSON is preserved with its SHA sidecar. The separate
`canonicalize.py` verifies that input and its source, binds all 200 judged rows
and all timing comparisons to the completed native report, and publishes the
canonical diagnostic. Use `inspection-canonical.json` with the repository's
native sealed-artifact loader. These scripts made no provider calls.

Canonicalization and cross-report verification passed in exec `6091d7`.
The original execution session **13811 exited zero** and its handoff completion
explicitly records `target_gate_passed: false`. Sessions 42892 and 48873 are
also terminal. No campaign process remains to poll or restart. Preserve the
frozen runtime, preflights, checkpoints and judgments for replay; implement
successors additively. The objective remains at least 95/100 with the required
latency on the same fresh full100 predictions.
