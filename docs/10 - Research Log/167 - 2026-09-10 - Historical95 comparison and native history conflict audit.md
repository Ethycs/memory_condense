# Historical95 comparison and native history conflict audit

**Date:** 2026-09-10  
**Status:** completed diagnostic; accuracy/latency target still unmet  
**Predecessor:** [166 - Full100 as-of result and reader failure audit](166%20-%202026-09-10%20-%20Full100%20as-of%20result%20and%20reader%20failure%20audit.md)

## What changed from 95/100

The historical policy scored 95/100 on the same validation questions and
approximately 1M-token pooled memories. It accumulated retrieval and answer
repairs and preserved prior predictions through its lineage. It did not prove
near-direct-API latency on those same fresh answers. The current summary-routing
path independently generated every timed answer and scored 80/100. These are
different pipelines; neither the memory size nor the question count explains
the difference. The historical result remains recorded, and the current fast
path has not matched its accuracy.

The authenticated latest paired comparison gives 78 both correct, 17 historical
only, 2 current only and 3 both incorrect. Of the 17 losses, 9 are multi-session,
3 temporal, 2 preference, 2 user lookup and 1 knowledge update. This supersedes
the earlier 95-versus-73 comparison for describing the latest accuracy gap;
both diagnostic artifacts remain preserved. It is not a causal ablation.

Current as-of total latency is 6.580/9.366 seconds median/p95, against
6.039/9.734 for the identical-evidence API and 5.083/6.353 for short API chat.
The matched-evidence allowance passes; the short-chat allowance and 95/100
accuracy requirement fail. Do not combine historical accuracy with new timings.

## Stronger reader on identical packets

`tools/evaluate_spine_reader_residual.py` selected all twenty completed as-of
misses for a diagnostic with Sol as reader, keeping every original message
verbatim. References and previous predictions were not reader inputs. The
reader retained the 256-token output limit and omitted temperature, matching
the original reader setting. Its transport was non-streaming batch rather
than the original timed streaming transport, so these are not serving latency
measurements. No Qwen calls or new retrieval occurred.

All twenty new reader responses sealed before any references opened for
judging. The fixed Sol judge then accepted four: ordinals 13, 50, 60 and 94.
There were twenty reader calls and twenty judge calls, with zero initial replay
hits. A subsequent complete replay returned twenty reader and twenty judge
hits, zero new calls, and the same sealed report. Session 73831 completed the
original execution; session 67441 completed the verified replay, both exit zero.
The earlier observer handle 97811 was unavailable after compaction; no original
provider execution was repeated to resolve it.

The four accepts do not establish four semantic improvements. At ordinal 60,
the new prediction `University of Melbourne.` was accepted as implicitly
supplying its location, while the prior `University of Melbourne` was rejected
for missing Australia. Preserve this grading inconsistency. Ordinal 94's
accepted 21-day answer still requires an event-grounding audit. Ordinal 13
abstains and ordinal 50 correctly chooses TikTok.

The same model reads and judges in this diagnostic, and the eighty previously
correct questions were not reanswered. **Do not claim 84/100, promote Sol, or
attach the previous serving latency to this result.** Stronger reading alone
did not resolve the residual set. Ten focused reader contract tests passed.

## Complete native-history membership audit

`tools/audit_spine_native_history.py` authenticated all ten source namespaces,
all 500 response/request chains and all 100 as-of judgment bindings. It checked
every hydrated slice against the sealed raw turn's text hash, role, date,
source and exact character coordinates. Native membership requires matching
the original session ID, date, role and complete normalized text; a record
ownership prefix alone cannot establish membership. Annotated coverage unions
overlapping intervals and duplicate copies rather than counting them twice.

Only the already-examined validation100 histories were inspected in the pinned
S dataset. The whole JSON container is parsed, but confirmation histories,
questions and answers are not retained or analyzed. No model or provider calls
were made. Gold annotations and source ownership are diagnostic labels and
must never enter production selection.

Results:

- Every original native turn is present in its pooled memory for all 100
  questions. This audit found no missing original history at ingestion.
- 96/100 final packets contain user turns outside the question's native
  history. There are 1,407 such selected user turns, versus 995 native user
  turns. These counts repeat a turn if selected by different queries.
- 97 questions have annotated support turns. All such turns are fully hydrated
  for 83 questions, including 12 of the 20 wrong answers. Turn coverage is not
  a proof of semantic sufficiency or correct interpretation.
- Six misses have incomplete annotated coverage: 53, 54, 61, 69, 86 and 87.
  The other two misses without full coverage, 13 and 42, are abstention cases
  with no annotated support turns.

Outside-native membership does not itself imply false or contradictory text.
The Sophia question supplies one concrete counterexample: the original history
states a coffee-shop meeting. The pooled memory additionally supplies a
different record's user statement about meeting Sophia at a grocery store;
that session is absent from the question's original history. Both complete
statements are in the final packet. Terra and Sol both choose the grocery
store. The two independently attributed occurrences are compatible with two
different people, but the pooled, unqualified question does not identify which
history's person it means. The retained gold answer assumes the original
history's coffee-shop meeting.

This is a corpus-construction ambiguity, in addition to ordinary routing and
reader errors. It is inherited from the existing pooled source databases, not
introduced by the latest summary ingest. Both old and current results use that
pooled scope, so the finding alone does not explain their accuracy difference
or invalidate every historical answer. Do not use the gold source prefix to
filter the memory, narrow it to the original ~100K history, relabel failures,
or claim that this would meet the 1M target.

The new audit's ten focused tests passed in 1.07 seconds. Full execution
completed in session 49506, exit zero. Replay in session 69310 completed with
`created: false` and a byte-identical report, exit zero. The script and its
pinned dependencies are now frozen with the artifact.

## Native long-history assessment, not a replacement score

The [official LongMemEval documentation](https://github.com/xiaowu0162/LongMemEval)
describes a coherent history for each question and provides a native M file
with roughly 500 sessions per history. It also records cleaning the histories
to prevent interference with answer correctness. That provides a concrete
alternative to pooling separate question histories as one user's memory;
it does not establish that every native M history is conflict-free or has the
required token count under this repository's tokenizer.

Public Hugging Face metadata was fetched without sending any local content.
The pinned repository revision is
`98d7416c24c778c2fee6e6f3006e7a073259d48f`.
`longmemeval_m_cleaned.json` is 2,737,100,077 bytes, LFS SHA-256
`9d79e5524794a2e6900a3aa9cb7d9152c5a3e8319c9a87c25494ba1eacee495f`.
The same metadata identifies the current S file by its already-pinned
`d6f21ea9...` SHA. Only S is present in the inspected local dataset directory.
The M content has not been downloaded, inspected or ingested. No benchmark
replacement or new answer campaign was executed.

The next data assessment should authenticate that exact public M file, inspect
only the locked validation100 histories, compare question/reference identities
and native evidence with S, and count actual tokens per complete history. Keep
confirmation200 sealed. If M's dates or golds differ, report the drift rather
than assuming the old paired identity survives. Any evaluated successor must
keep complete >=1M-token histories, separate history boundaries and the same
fresh accuracy/latency requirements. Preserve the existing pooled campaign as
its own stress test; do not substitute a new score into the historical series.
Determine ingestion reuse and cost before starting another large summary job.

Independently, the current corpus still has clear coverage failures (missing
plant purchases, smoker, furniture, clothing, trips and classes) and reader
failures with present evidence. A generic successor may improve those, but
must be checked on all validation questions, without question-specific rules
or selecting from old accepted answers. Parent-summary routing and a proven
query-time attention gain remain unestablished.

## Sealed artifacts

Paths below are relative to this worktree's `eval_results/`.

| Artifact | SHA-256 |
| --- | --- |
| `full1m-spine-sol-reader-residual20-20260910-r1/preflight.json` | `b3359ba8d7f04ef004580c26b533e68851d8459e020ce1d60d2f92a3c4f29c51` |
| Same root, `answers.json` | `e715dd5d90f1328ee37944672d93e4c0d6f4e88dea3d501a879db18a6cc09414` |
| Same root, `report.json` | `a567fe32caa716fa2e607e3af4037d3aaa2ca764a62438daf27991cd5b6fecc5` |
| Same root, `complete.json` | `5f0f4bd123d0c1252ebcd04dc8c77f844a73f7cb962552f40b73d155cfd2f541` |
| `full1m-spine-native-history-audit-20260910-r1/audit.json` | `67145996687a865ab45e1df82e5af93e67f96015b25f657da85a555205f93fdd` |
| Same root, `historical95-fresh80.json` | `dd74fa288f12ff8925f8a08b96b4c010a0c39dd98c6272f18bb912b559818095` |
| `native-longmemeval-m-assessment-20260910-r1/file-metadata.json` | `b3c62b94f93c7d397c2d0de3b909fe78d48812e9d5b67b54bc51b7f234d4f622` |

The current goal remains active. All provider and diagnostic processes named
above are terminal; no old job remains to restart. Preserve frozen runtime,
responses and judgments. No confirmation evaluation or fresh full100 successor
was launched in this continuation.
