# Memory evaluation lifecycle boundary

**Status:** Cached-memory lifecycle verified; full application lifecycle unverified.
**Date:** 2026-09-15.
**Scope:** One 1,098,417-token history and the unchanged 100-question set.
**Depends on:** [Research Log 221](221%20-%202026-09-15%20-%20Dense%20seeded%20parent%20context%20on%20one%20history.md).

**Subsequent resolution:** Research Log 223 verifies normal application ingestion
and independent reopening; Research Log 224 records all 100 live answers through
that persisted application. The gap described below is the historical assessment
that led to that integration, not the current execution state.

## What the current result measures

The latest completed candidate remains **84/100**, with a 3.985-second warm
median and a 7.328-second p95. Its API control median is 3.869 seconds. The
95% target is unmet. The questions are source-grounded generated development
questions, not an official LongMemEval result.

The runner uses completed raw-body, atomic-summary, attention-derived hierarchy,
and summary-vector artifacts. `evaluate_native_spine_design_slice.load_namespace`
validates those source bindings and materializes the one complete namespace.
`ResidentNativeSpineContextMemory` receives the summary index, stored hierarchy,
encoder, and raw-turn loader. No question or reference is supplied to that
materializer.

For each question, `ResidentNativeSpineMemory.retrieve` embeds the query, routes
over summaries, expands through the stored hierarchy, and hydrates selected
original spans. The candidate repeats retrieval inside the answer timer. The
matched API control receives the same prompt without timed retrieval. The
candidate therefore does not answer directly from an oracle-selected source
conversation or a reference answer.

All 100 candidate answers and 100 controls must be complete before the judge
opens the separately stored references. The raw audit independently reconstructs
each served excerpt from the original source body and occurrence. Ingestion and
the 37.253-second cold resident load are excluded from warm answer latency.

## Gap in application lifecycle coverage

The evaluation does **not** call the normal application's
`IngestWorkflowMixin.ingest` / `MemoryCondenser.ingest`, persist native-spine
memory through that application, close it, and reopen it for the questions.
It constructs the resident native memory directly from compiled artifacts.
Thus the measured boundary is **compiled ingestion artifacts → resident memory
retrieval → answer → grading**. The application ingest entry point and its
persistence/restart integration remain unverified by these 100-question runs.

The prepared conversation-grouping runner now records that boundary explicitly
in its preflight and report. Its new comparison has not been launched. Original
completed reports and scores remain unchanged. The full application lifecycle
claim requires an integration check through the actual supported ingest and
reopen path; another packet-layout score alone cannot establish it. That work
must retain one history and 100 questions, with ingest measured separately from
warm answering. It does not justify returning to 100 histories.

## Verification performed

The latest preflight, question/scope bindings, all 200 response journals, report,
and completion bindings were revalidated. The preflight specifies one history,
one namespace load, 100 questions, zero new history compilations, fresh timed
retrieval, and no reference loading during answering. Code inspection verifies
that reference opening follows the complete-answer seal.

The existing raw audit was rerun without an encoder or provider. It reproduced
the same receipt for **100 packets and 2,656 exact source spans**:
`7117e6fa6b71779d268c0a9ec663ad8f3384554ff59caff913cfb6e2ab7c83d8`.

```powershell
.\.pixi\envs\dev\python.exe -X utf8 -m tools.evaluate_native_spine_context100 audit --root eval_results/native-spine-context100-dense2048-20260915-r1
```

Sixteen focused presentation, context-policy, and renderer checks passed in
2.54 seconds. These establish exact-span preservation, source-order validation,
future-source exclusion, and packet-budget behavior; they do not close the
application ingest/restart gap.

```powershell
.\.pixi\envs\dev\python.exe -X utf8 -m pytest tests/test_native_spine_threaded_presentation.py tests/test_threaded_section_context.py tests/test_native_spine_context_policy.py -q --basetemp=.tmp/pytest-threaded100-20260915-r1
```
