# Scheduled complete-memory compilation

**Date:** 2026-09-10  
**Status:** four complete-memory workers prepared; scheduler running behind raw ingestion

The remaining full100 preparation now advances automatically as each whole raw
namespace completes. The raw scheduler from Log 156 remains the sole owner of
ingestion. A separate serial scheduler audits, admits, compiles, and indexes
each completed memory, then prepares its fifty evaluation requests. It sends
no benchmark answers or judgments. The latest measured scores remain 41/50 for
the existing reader and 39/50 for both tested successors; semantic-seed answer
accuracy and latency remain unmeasured.

The earlier 95/100 still belongs to the cumulative retrieval and answer-repair
policy on approximately 1M-token memories. Its final result preserves verified
prior answers and does not establish fresh API-like end-to-end response time.
Logs 102 and 132 describe that distinction and the authenticated historical
95-versus-73 comparison. This continuation changes preparation scheduling only.

## Implementation and execution boundary

`tools/finish_spine_memory_namespace.py` requires the exact raw completion for
every prepared request in one of offsets 060/070/080/090. It audits all source
summaries and stops on unresolved schema failures. Every and only oversized
summary is eligible for the existing summary-only compaction method: eight
jobs per batch, complete all original batches, preserve valid outputs, then
allow at most two recovery calls per invalid slot. Source admission and v10
verification must reproduce the same method as the first six memories.

The worker then builds attention-partitioned leaves with at most 64 new Qwen
calls, followed by semantic, user-summary, and passage indexes. Each compiler
runs in a separate child process so Qwen memory is released before BGE loads.
The existing leaf and passage policies must match the frozen experiment. It
prepares fifty unstarted requests and publishes the namespace binding into the
r2 full100 campaign. It does not run the evaluator's answer or judge phases.

`tools/schedule_spine_memory_completion.py` binds the workers to the actual raw
scheduler release, including its PID and creation time. It waits for a complete
payload and digest sidecar, verifies the raw completion, and runs only one
memory worker at a time. An incomplete namespace, corrupt completion, failed
raw dependency, reused PID, or expired twelve-hour wait cannot trigger work.
Failures preserve reservations and stop subsequent workers without retries.
When all four workers finish, the scheduler prepares the full500-request
runner; timed execution still requires an idle worktree and fresh readiness.

The raw scheduler may continue while memory compilation runs. No timed answer
comparison may run alongside either process. The scheduler does not execute
raw requests or create another ingest owner.

## Frozen plans and observed execution

Root: `eval_results/full100-spine-memory-completion-20260910-r1`.

| Plan | SHA-256 |
| --- | --- |
| Schedule | `0c515019bc3fecd4e8ac3dc79af4c31dd27c0ba57b18d3c870db508f5af7c71d` |
| Offset 060 worker | `8c445701697c732e70cbb113b37387df64b50342bb8f1adadb8edabd2fe8c370` |
| Offset 070 worker | `b0ef30bfebc2f82c97b526377731ff1f635767d2a3ba301795715cddeec2dc0b` |
| Offset 080 worker | `55c07ab4883e6cc4d1a32fb65696ba8b0f52ee5a6661a99b77cc25351a742cba` |
| Offset 090 worker | `b5576eb10c3298ed755dd6f16ec305bf1190cedfbc239eb9c8b112ea371e1e67` |

The preparation completed without provider calls. The authorized execution is
session **86943**, PID **44128**, process creation time **1789053794.025061**,
released at **15:23:15 UTC**. Release SHA:
`cab8adc6aff1182ba41f1eafe61073e7edc13fb7001ad38adf38faed0025ae48`.
It initially reported waiting for complete offset-060 raw input.

The existing raw session **42892**, PID **63016**, creation time
**1789052354.6007824**, was independently observed still running. Its release
remains `073f4813e0cd9d4a3f5ef5a752af37f529c4659028606cc50144d454402a549a`.
At the subsequent inspection, offset 060 had **703 responses / 707 requests**
out of 792, including the 403 preserved responses. Offsets 070/080/090 were
still unstarted. No seventh-memory source admission or index completion is
claimed at this checkpoint.

## Validation and continuation

The two focused suites pass **14 tests in 3.38 seconds** (exec chunk `468fa0`).
They cover serial ordering, raw completion publication, missing/failed/terminal
dependencies, reused process IDs, corrupt artifacts, bounded waits, source
schema rejection, admission incompatibility, bounded compaction recovery,
failure preservation, duplicate release rejection, and absence of answer/judge
execution. These are orchestration tests with simulated compiler outputs;
actual seventh-memory compilation is still pending.

Two initial test invocations failed during temporary-directory setup: the
default external pytest root denied access, then a new workspace base lacked
its parent directory. A unique workspace base with its parent created ran the
same tests successfully. No implementation changes or escalations were needed
for those setup failures.

Continue by observing both existing sessions. Do not launch a duplicate worker
or ingest scheduler. Once compilation completes, all 500 fresh answers must
finish before any reference-based judging. The frozen accuracy and both API
latency requirements remain unchanged, and confirmation remains unopened by
this continuation.
