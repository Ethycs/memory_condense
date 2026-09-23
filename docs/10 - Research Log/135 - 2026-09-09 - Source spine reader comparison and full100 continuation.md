# Source spine reader comparison and full100 continuation

**Date:** 2026-09-09  
**Status:** reader comparison sealed at 9/10 versus 9/10; joint target unmet  
**Predecessor:** [134 - Additive user spine routing on complete memory](134%20-%202026-09-09%20-%20Additive%20user%20spine%20routing%20on%20complete%20memory.md)

The general reader-policy change recovered descriptive identification but did
not improve the aggregate development score: both policies scored 9/10 on
identical evidence from the complete 1,041,276-token memory. The new reader's
median total time was 6.401 seconds, against 5.516 seconds for its short API
control. Its p95 was 11.873 seconds against 9.048 seconds. Accuracy and latency
still need to pass together on all 100 questions across ten complete memories.

## Why the historical 95/100 still stands

The earlier `policy-v5-r3` validation score is 95/100 over the same locked
100-question population, using the cumulative evidence and answer-repair
pipeline. It reused authenticated prior answers and judgments; it was not a
fresh timing run of all 100 answers under the current latency requirement.
The faster historical user-spine pipeline scored 73/100. The authenticated
comparison found 72 shared correct answers, 23 lost by the fast pipeline, one
fast-only gain, and four shared misses. Most losses involved multi-session or
temporal reasoning. These results compare different pipelines and do not
isolate a causal contribution from any one component.

The newest 9/10 is a development result from the summary-routing successor,
not a replacement full100 score. Retaining the old accuracy while reducing
live query work is the active engineering objective. See
[Research Log 132](132%20-%202026-09-09%20-%20Joint%201M%20accuracy%20and%20latency%20target.md)
and the [95 percent campaign playbook](../02%20-%20Implementation/13%20-%2095%20Percent%20Full100%20Campaign%20Playbook.md).
Confirmation200 remains unopened.

## Matched reader comparison

`eval/spine_reader_policy.py` adds general instructions to return a supported
identifying description when a proper name is absent and to make relevant
preferences and compatibility explicit in recommendations. It contains no
benchmark answers, names, source IDs, or question-specific selectors.

`tools/evaluate_spine_reader.py` keeps both policies on identical source-spine
evidence with the existing 3,072-token context budget. Each policy has its own
short API and identical-evidence API controls, for six arms and 60 streamed
Terra calls. Memory arms perform embedding, routing, and exact hydration inside
the timer; there is no query or answer cache. Memory/API pairs are adjacent,
their order is counterbalanced, and group order rotates. The same streamed
memory answers receive independent Sol judgments.

Root: `eval_results/full1m-spine-reader-joint-offset000-20260909-r1`.

| Artifact | SHA-256 |
| --- | --- |
| Preflight | `a0ce7f887ff8f93894c3d67acf81ed7ea6899b41e9388deb54a6bbe12d914f32` |
| Answers | `051a3761c5e3fe4a51b1a3f3b0fc43e6caacf3298c60f2bd65fff2d8872ec083` |
| Joint report | `c2a0263fba0433757bbbf1cf3132065f7218889bbc6838048a8fb9357e99c95c` |

All 60 answer calls completed. Twelve physical Sol calls supplied 20 logical
judgments; replay used 12 authenticated hits, zero calls, and reproduced the
report. No bulk provider ingest overlapped the answer timing. Provider-free
preparation for the next namespace overlapped some early answer calls.

| Arm | Accuracy | Preparation median | Total median | Total p95 |
| --- | ---: | ---: | ---: | ---: |
| Base short API | Not scored | <0.001 s | 6.285 s | 9.487 s |
| Base memory | 9/10 | 0.219 s | 6.953 s | 12.083 s |
| Base identical-evidence API | Not scored | <0.001 s | 6.443 s | 8.935 s |
| Reader short API | Not scored | <0.001 s | 5.516 s | 9.048 s |
| Reader memory | 9/10 | 0.234 s | 6.401 s | 11.873 s |
| Reader identical-evidence API | Not scored | <0.001 s | 6.955 s | 8.408 s |

The gateway delivered one visible content event per answer, so visible TTFT
was essentially total response time. Reader memory was 16% above its short
API median and 31% above p95. It also exceeded its identical-evidence API p95
by 41%. This run fails the provisional 10% latency allowance.

The reader recovered the bluegrass/banjo identifying description. Its only
failed judgment was the photography recommendation. Both answers named a
Sony-compatible Godox V1 flash, but the base passed and the reader failed.
The judge requested more explicit preference and quality constraints. The
answers are similar enough that this is evidence of grading sensitivity, not
a sound basis for manually changing the score to 10/10. Preserve the recorded
9/10; do not selectively rejudge or combine the best answers across runs.

## Prepared reader v2

`eval/spine_reader_policy_v2.py` extends the first reader policy with two short
recommendation sentences: state supported user preferences and compatibility,
then suggest suitable options. The versioned harness
`tools/evaluate_spine_reader_v2.py` uses reader v1 as its base. Evidence,
questions, retrieval, and model remain fixed; only the system instruction
changes. Each policy again has its own two API controls.

Root: `eval_results/full1m-spine-reader-v2-joint-offset000-20260909-r1`.
Preflight SHA:
`6fbd4b0d82c8afd671495d406893c424bf2fadf9c8ea79737a5cc910e70a9e42`.
Evidence-equivalence SHA:
`7a112350c9398e039da0eec4a7d29ea0207c53f10d484e269d8b10ec772429cb`.
There are zero answer reservations and no answers artifact. Run the prepared
60-call comparison only when bulk provider work is quiescent, to reduce
contention in the latency measurement. Do not modify its frozen dependencies.
The later overflow comparison in
[Research Log 136](136%20-%202026-09-09%20-%20Preserve%20selected%20user%20evidence%20before%20attached%20context.md)
uses this same reader-v2 policy on both arms and takes execution priority.
This six-arm preparation remains unexecuted.

## Complete-memory continuation

Session **85650** is executing namespace offset 10 under the existing
801-request preflight, four concurrent raw Terra calls and zero retries.
Preflight: `eval_results/full100-spine-corpus-20260909-r1/offset-010/execution-prefix-0801.json`,
SHA `d032520098a0302f762935a5f1844b1cb0133bd16a4c1235cc64533a169ce1bf`.
The process is still active; outputs include completed batches through the
330s in non-sequential completion order. Preserve its authenticated journals
and do not restart an unacknowledged request.

The original executor continues to print strict support-quote failures.
Those diagnostics are separate from current source-bound routing admission.
`tools/audit_spine_source_admission.py` replayed the first 120 completed
requests under the current admission checks: **zero failed batches, zero
oversized summaries, zero schema failures, and zero provider calls**.
Audit: `offset-010/source-admission-audit-prefix-0120.json`, SHA
`0e8bdb4f707b1d1106ae4c6891ef9277c162a5c9fc85f04990ef9eba319dbdb8`.
Exact input-fragment ownership is verified; summary entailment is not.
The later reader's exact raw evidence remains the factual authority.

A subsequent audit of the first **200** complete requests also found zero
failed batches, oversized summaries, or schema failures, with zero calls.
Artifact: `offset-010/source-admission-audit-prefix-0200.json`, SHA
`aae5b24d7bd4092d85857e372d8ee003bc016a5d89429954f3cc0d2d03d63ff3`.

The other eight namespace execution preflights were prepared, covering
6,654 raw requests with zero calls at preparation. Readiness artifact:
`eval_results/full100-spine-corpus-20260909-r1/remaining-execution-preflights.json`,
SHA `d0d32a473bbd1c66a5a4030a75a75e57caca2cd674af7b8e2f9b5955d7ee46a4`.
The request counts at offsets 20 through 90 are 817, 838, 837, 834, 792, 838,
830, and 868. All populations were prepared without gold or relevance filters.

Offset 20 subsequently started in session **91664**, using its already sealed
817-request execution preflight at SHA
`864547f5bab6400f323448007dbb54fe1b9c3449610c7d23d927a04def5e1cac`.
The process is confirmed live and producing completed responses. Together the
two raw-ingest jobs permit at most eight concurrent requests, four per memory,
with zero retries. No latency measurement runs during this overlap. The other
seven namespaces have not started provider execution.

Execution handoff: `parallel-namespace-ingest-offset010-020.json`, SHA
`4f179f151c96df76b3685d7563506604442537f9ca1c3d87977a685c89f7092b`.
This records the two live jobs; their individual preflights preceded I/O.
The first eight offset-20 requests pass current source-bound admission without
repairs or new calls. Audit: `offset-020/source-admission-audit-prefix-0008.json`,
SHA `38b4f8cf2de207e65cf4ca431bddedb4c601008f6dd24c160e067cc3e11e0d8d`.

Later completed-prefix audits found only bounded summary-length failures:
offset 10's first 360 responses contain one 129-token summary, while offset
20's first 120 contain two summaries of 132 and 133 tokens. Neither audit found
schema or attribution failures, and both made zero calls. Their SHA values are
`29bfe1c8989353c1b81e617f79b2052eeae3a94f810b516f7fd99743fd0cb242`
and `582fcf862ef352f763ed3f3db4c868d5c609d3258f405de924144bc463919d77`,
at `offset-010/source-admission-audit-prefix-0360.json` and
`offset-020/source-admission-audit-prefix-0120.json`, respectively. Keep these
failures until the complete response populations are available, then prepare
the full-namespace summary-only compaction batches. No such compaction call has
run for either live namespace. If more than eight summaries require repair,
the existing single-batch repair tool needs a versioned successor; do not drop
failures to fit its cap.

After the live job finishes, audit all 801 responses. Any over-budget summary
may use the existing authenticated summary-only Qwen compaction path; resolve
schema or attribution failures explicitly. Admit the complete atom population,
compile its attention-guided leaf projection, compile the semantic index, and
compile user-summary addresses before preparing its evaluation. Query routing
currently uses summary embeddings and exact source-spine hydration; Qwen is
used for summary compilation and attention-guided boundaries at ingest.

`tools/compile_spine_user_addresses.py` now compiles user-summary vectors
without an evaluation preflight, questions, or raw text. On offset 0, it
reproduced the earlier user-summary matrix and address identity exactly.
Comparison SHA:
`8ec95fc0833248de5f72aa50b5d8487545b96fead94d9d17db807f5765cc64e4`.
The prepared reader v2 evaluation retains its original bound addresses.

The full100 aggregators for the source-spine and reader harnesses distinguish
namespace-specific admission receipts from the admission method itself.
`tools/spine_admission_policy.py` authenticates the namespace bindings before
excluding their receipt hashes from method comparison. Actual rules and code
remain part of method identity. The v3 policy without compaction and v4 policy
with compaction remain distinct legacy receipts. A subsequent replay verifier,
described below, establishes the common conditional rule without changing those
receipts or requiring unnecessary compaction. No full100 aggregator has run.

## Conditional admission verified by replay

`tools/verify_spine_admission_method.py` now recognizes only the exact v3 and v4
receipt templates generated by the current admission implementation. It
reconstructs each compaction input from the original attributed Terra summary,
checks that the Qwen request contains that summary with its correct role and
timestamp, and replays the original admission tool against every raw response.
The replay must reproduce the sealed atoms and quote diagnostics byte for byte.
All summaries at or below 128 tokens remain unchanged; every over-budget
summary must have its authenticated compaction. Missing or changed evidence,
arbitrary summary edits, changed admission rules, and foreign repair roots fail.
The existing compaction protocol remains one batch of at most eight summaries.

The three source-spine/reader full100 aggregators use this replay verifier and
record its artifact hash for each namespace. A stored success flag is
insufficient: aggregation replays the verification again with no provider
client. Complete memory, all ten namespaces, one query/reader method, and both
accuracy and latency gates remain required. A partial admission certificate
cannot supply a full100 method.

Real-data checks completed with zero new calls:

| Population | Replayed raw responses | Atoms | Compacted summaries | Complete memory |
| --- | ---: | ---: | ---: | --- |
| Offset 0 | 850 | 5,556 | 4 | Yes |
| Offset 10 prefix | 200 | 1,307 | 0 | No |

The first memory retains atom SHA
`6444cf274d60e025793c70e60f0d41de4766efa77b6b98950b8dabd0dfb889a5`.
Its verification SHA is
`e48f69e592c8ff77ded586782ee92a971f0c8599c9ef23e5b805e72166aa08de`.
The second memory's prefix retains atom SHA
`a4d2ee156738b6da2715cc4965adbc76e3f89b22e003428f96c7ba684e57d446`.
Its verification SHA is
`415d22fdfa68a7324fd01827dee7c3de4151226342839890033ff00f3740df64`.
Both certify conditional method SHA
`55e686613eff513b3029f7c75069d3d5be7683570b7fecc1795b32f8029a190f`.
Comparison artifact:
`eval_results/full100-spine-corpus-20260909-r1/conditional-admission-method-comparison.json`,
SHA `3527956378733006ba7b02e0cfa3e4238ee6b9374d1915141249aa4ff3b6e2fa`.

After each complete namespace is admitted, run the verifier before aggregation:

```powershell
.\.pixi\envs\dev\python.exe -X utf8 -m tools.verify_spine_admission_method --atoms <complete-source-bound-atoms.json>
```

Add `--summary-repair-root <bound-repair-root>` exactly when the admission receipt
binds compactions. This check performs no model calls and does not alter summary
or raw evidence content. The prepared reader v2 preflight was revalidated at
its original SHA; its answer execution is still pending the two raw-ingest jobs.

## Verification

The reader lifecycle tests passed eight cases across both policy versions,
covering live retrieval, identical raw evidence, policy-matched short controls,
counterbalanced calls, changed-prompt rejection, and interrupted-request/gold
guards. The admission-policy and full100 gate checks passed seven tests,
including rejection of partial populations and accuracy/latency passing on
different methods. The admission-audit test passed one case covering all
oversized summaries and malformed attribution. These are implementation checks,
not accuracy results. The current documentation continuation also authenticated
the sealed result and prepared successor hashes without provider calls.

The conditional admission continuation passed 14 focused tests, including
real checkpoint replay of both conditional branches, refusal of a changed
summary even after its artifact hash is recomputed, partial-memory exclusion,
changed-policy rejection, and the full100 gates for both reader versions.
The completed 1M-token memory also passed the real replay described above.
