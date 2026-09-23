# Multi-batch admission and third complete memory

**Date:** 2026-09-09 (local; execution observations are September 10 UTC)  
**Status:** three complete indexed memories; third comparison finished, results in Log 141  
**Predecessor:** [139 - Two complete memory overflow comparisons](139%20-%202026-09-09%20-%20Two%20complete%20memory%20overflow%20comparisons.md)

The third memory now contains **5,357 admitted source-bound atoms**, covering
**479 conversations and 1,045,527 raw token proxies**. Every prepared raw
fragment is retained in its original order. Admission and verification replay
all 817 saved raw responses with zero provider calls. The latest answer result
remains 18/20 for the overflow candidate versus 16/20 for its controls; this
ingest completion does not add an accuracy result or meet the full100 target.

## Complete compaction population

The complete audit found nine over-budget summaries, with no schema failures.
`tools/repair_spine_summary_budget_v2.py` reconstructs the exact generated
summaries from their authenticated Terra responses and packs all jobs in raw
request order, with at most eight jobs and 7,000 prompt tokens per Qwen request.
It reconstructs the preflight again before allowing calls. Resealing a changed
job cannot introduce foreign input. Qwen receives generated summaries with role
and date attribution; raw fragments remain outside Qwen.

Two requests ran. The first produced eight valid summaries; the second returned
one summary that still exceeded 128 tokens. Both responses remain saved in the
original root, and no aggregate repair artifact was published there.

`tools/recover_spine_summary_budget.py` reuses the hierarchy compiler's bounded
single-job recovery: at most two calls, asking for 48 and then 24 words, with
the same 128-token validation limit. It retains every valid original slot.
Only the failed ninth summary required recovery, and the first recovery call
succeeded. The final aggregate contains all nine repairs. Total physical Qwen
work was **three calls: two original batches plus one recovery**. No transport
retry or replacement of a valid compaction was used.

Original compaction root:
`eval_results/full1m-spine-budget-repair-offset020-20260910-r1`.
Preflight: `bbfc80e0e25737ad83f488d6b5e449c6591c7bef2ec9899872c32b2545aae873`.

Recovery root:
`eval_results/full1m-spine-budget-recovery-offset020-20260910-r1`.
Preflight: `57112d91e747cbf0307806c32cb6b32df4dca7fa4636403eb98d33766da05679`.
Repairs: `dc49eac947437558c59d21ce8554f7b0eda53fd53974580c0be9c3bd7d16d58f`.
Replay uses two original-batch hits and one recovery hit, with zero calls.

## Admission and comparison rules

`tools/admit_spine_corpus_v2.py` adds source policy v5 for deterministic batches.
`tools/admit_spine_corpus_v3.py` adds policy v6 for preserved valid slots plus
bounded recovery. Both retain the existing exact source binding, unchanged
quote diagnostics, and prohibition on using generated summaries as answer
evidence. All repairs must apply exactly once. Neither claims summary entailment.

Version-3 verification reconstructs the entire oversized-summary population
from every original raw response before comparing batch identities. Version 4
adds the completed-invalid-slot recovery branch and counts its attempts. The
earlier tools, response journals, source atoms and certificates remain unchanged.
Use `tools/report_joint_source_spine_overflow_full100_v4.py` for eventual full100
aggregation; its complete-population, accuracy and matched-latency requirements
are unchanged.

All three complete memories replay under common method SHA:
`070dde34f28dce3d0326540485076a3e031934d59ff27ca51184d486e95cbea6`.

| Memory | Version-4 verification SHA | Compaction provider attempts |
| --- | --- | ---: |
| Offset 0 | `439a7dade94662cf34e9cebe580348fb08e6a71e7c14751dc6423c604a870fe2` | 1 |
| Offset 10 | `c3494e61d50767fdcefb5ae3d851f2fa546a0cf540175ccc87e03a3a1d91b071` | 2 |
| Offset 20 | `d0a81715473c0778fdc4514a1d6031f0adac4bfaabcc9ca79f2441ee681d5223` | 3 |

Offset 10 includes its previously recorded unknown transport attempt and one
successful successor. Offset 20 includes two completed original batches and
one validated recovery. Its separate five raw transport reissues remain bound
to the original recovery inventory. Different attempt counts are reported,
not erased to make the methods comparable.

Offset-20 atoms:
`eval_results/spine-transport-recovery-20260910-r1/corpus/offset-020/source-bound-atoms-prefix-0817.json`,
SHA `e22587c5809c98c15310bfeb72270febd9d5263cc2aa87d713b20c6c009fef33`.
There are 2,139 atoms with retained quote diagnostics and nine compacted atoms.

## Verification and next execution

Later result: the third comparison finished at 7/10 for both arms, bringing
the candidate to 25/30 versus 23/30 for its controls. Judging replayed with
zero calls. The recorded running-session notes below are historical; current
results and the live fourth-memory ingest are in
[141 - Development30 result and omitted user evidence](141%20-%202026-09-09%20-%20Development30%20result%20and%20omitted%20user%20evidence.md).

The focused suite passes **28 tests**, covering nine jobs across two batches,
exact replay, missing and unacknowledged responses, invalid labels and omitted
jobs, changed summary inputs, unchanged valid outputs, bounded recovery
exhaustion, common methods across legacy and recovered admission, and the
unchanged full100 accuracy-plus-latency gate. New modules parse, and the edited
gate-test file passes `git diff --check`.

Third leaf root: `eval_results/full1m-spine-leaves-offset020-20260910-r1`.
Preflight: `6693f5068b51436f3461081bb369030776c428eb0fc7e703d5c1d1bbba58315e`.
Its method matches the earlier two leaf compilations exactly:
`06c51b3892c2ecebc3155003b3ec6906d62b12eb6f55b7095eb571787f94e878`.
The compiler finished within its 32-call ceiling, using **seven** Qwen summary
calls. It produced **2,693 leaves across 479 sources**, using local summary-only
Qwen attention and deferring parent summaries unused by the measured query
arms. Hierarchy SHA:
`b71235db6564142dead625a5ff10b2250d0ab33a0ed840a22c5a4d9e59032c31`.
Semantic index root: `eval_results/full1m-spine-semantic-offset020-20260910-r1`,
SHA `35af2f3d8751195c46152f76ceadcb30fec72c29b10e8e55d3839e553da5aa59`.
User-address root: `eval_results/full1m-spine-user-addresses-offset020-20260910-r1`,
SHA `3da5c0d0459315e98e3370232e5c80f2cce14a9dc554b62efb02895ad474ab58`.
All three semantic indexes have the same embedding identity and hierarchy
method. Compilation is complete, with no new provider calls in either vector
stage.

The unchanged 50-call matched comparison for ordinals 20–29 is running in
session 65824. Root:
`eval_results/full1m-source-spine-overflow-joint-offset020-20260910-r1`.
Preflight: `9297b13623172f2f2543e9896be8b8de71a2555aea561656aed4538a54a9f5ef`.
Every non-namespace answer-policy field matches the first two comparisons.
Defer bulk provider work and GPU compilation throughout answer timing, then
judge this complete answer population and replay the report with zero calls.

The next raw namespace, offset 30, has a freshly replayed zero-call execution
preflight `583cf02779d46eda972f0e82f81853dd91f9713336b42554ff86eb3c49da83d1`
for 838 requests. Seven raw namespaces remain prepared but unstarted;
confirmation200 remains unopened.
