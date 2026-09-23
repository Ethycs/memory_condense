# Preserved recovery execution and resumed reader comparison

**Date:** 2026-09-10  
**Status:** reader comparison subsequently completed; see Log 148 for results  
**Predecessor:** [146 - Fifth memory gateway timeout and preserved recovery](146%20-%202026-09-10%20-%20Fifth%20memory%20gateway%20timeout%20and%20preserved%20recovery.md)

**Continuation:** [Log 148](148%20-%202026-09-10%20-%20Reader%20development40%20result%20and%20summary%20term%20coverage.md)
records the completed 33/40 versus 32/40 result, zero-call replays and resumed
fifth-memory ingest. The running-state notes below describe the earlier checkpoint.

At that checkpoint the measured result remained **26/30 versus 25/30** from Log 144. The earlier
95/100 already used approximately 1M-token memories and the same locked
question identities, but came from the cumulative retrieval and answer-repair
pipeline without a fresh demonstration of API-like latency. Neither result
establishes the active joint accuracy and latency target. Confirmation200 is
still unopened.

## Recovery execution and admission

The new `tools/execute_spine_transport_recovery_v2.py` and
`tools/spine_transport_lineage_v2.py` support the staged offset-40 namespace
without relabeling the old offset-10/20 recovery. The executor preserves all
99 completed responses, declares 733 first attempts plus five additional
attempts for unresolved original requests, permits a single release after
fresh successful readiness, and preserves any failed successor reservations.
It does not retry an unacknowledged successor or release a partial run again.

The verifier authenticates the original five request protocols using a checked
successor runtime context. Instantiating a runtime on the unresolved original
directory correctly refuses replay, so the first test run exposed and then
resolved that verifier error. No original reservation was altered to make the
check pass. Legacy offset-10/20 transport receipts remain identical.

The real executor preflight is sealed under
`eval_results/spine-transport-recovery-offset040-20260910-r1/execution-raw-v2.json`:
`f512f9bba2e2654a0bbb907e2cf9d53dd65f8ce7c0d665f527304d57a4f5ef57`.
Preparation replayed the 99 retained responses and confirmed all 738 future
slots remain unstarted. This preflight freezes the executor and verifier
implementation; no recovery provider call has been sent at this checkpoint.

Version-7 admission adds the new transport accounting without changing source
summary semantics, syntax repairs, or exact raw support requirements. Both
the passage full100 report and reader-v3 full100 report have successor entry
points using it. They retain the full100 >=95% gate and the provisional 1.10
median/p95 TTFT and total-latency gates against both API controls.

All four completed memories replay under one version-7 admission method,
with their original atom bytes unchanged and zero model calls:

- Method: `b090b92277e3dd392deed41825050bf44343f15d1540bb7c940c9cbc8baa0c30`.
- Four-memory aggregate: `14f965a7d11be175c243ebff7d1b8ad9eeb8370236c48537196000c3b9c4d449`.
- Offset 0 verification: `8f43c69acf1a96069a5ae68143cae267d55d8afeccd7e0ec6343b901f9f6765f`.
- Offset 10: `59e2d74141c633f735a6d3af69675b53c50565fe82a127cc5bfb441b7a4f3224`.
- Offset 20: `a14d5cb86c1a6da8fe4e919b9ac43dc546561c2dba1a098ccf22f5cfb96d69ac`.
- Offset 30: `875122b26fe3817feaae3cd51af5b47948f25782d11427a3c466520fe356e7d1`.

Aggregate path:
`eval_results/full1m-source-spine-facets-development30-20260910-r1/admission-four-memories-v7.json`.

## Readiness and comparison scheduling

A second readiness observation still failed with HTTP 500 for both models:
`eval_results/spine-gateway-readiness-offset040-20260910-r2/report.json`,
SHA `b1e84783ac332991843037bdc2cc5fa47b874af3b3251b06822002efb2879eba`.
The third observation succeeded for both models at **08:54:22 and 08:54:25 UTC**:
`eval_results/spine-gateway-readiness-offset040-20260910-r3/report.json`,
SHA `b6305459813224640f12935e85de638ce676bcf357e97bf9a068e7a60ad4ac30`.
Each observation uses two bounded synthetic probes with zero retries. No
benchmark or raw corpus content is sent by the readiness probes.

`tools/run_spine_reader_after_timeout.py` binds the original stopped runner,
its terminal dependency observation, and the original 240 prepared requests.
It changes only the scheduling dependency: the four complete memories can be
evaluated without waiting for the fifth namespace. It rejects changed prompts,
partial populations, duplicate releases and other Python jobs in this
worktree before starting. Answer/judge order, exact evidence, reader policies,
budgets and timed concurrency remain those of Log 145.

Successor scheduler root:
`eval_results/full1m-spine-reader-v3-development40-timeout-successor-20260910-r1`.
Preflight SHA:
`8bc7c687a3f0fa693102f9414b740446f15ae2ab6462b5952a1a40788140d93e`.
It started using the successful third readiness observation in session
**62889**. No bulk ingest, GPU compilation or large replay runs alongside this
timed comparison. Keep that exclusion until the runner is terminal.

## Checks and continuation

The new transport tests pass **9/9**; the associated legacy transport,
staging, admission and full100 gate checks pass **105/105**. The six scheduler
checks pass, including failed readiness, concurrent work, partial completion,
duplicate release, exact serial population and stopping before judges after an
answer failure. These are **120 passing checks** across the three focused
runs; they are not accuracy evidence.

Poll session 62889 and preserve every original reader reservation. On success,
the original development40 report is built from all four complete memories
and the judge populations replay without calls. On failure, retain the
successor's failure and release receipts; do not restart it blindly. Any
development score remains separate from full100 and unopened confirmation.

After the timed run is terminal, a fresh successful readiness observation is
required before releasing offset-40 raw recovery. The prior readiness report
expires after five minutes. Offsets 50 through 90 remain prepared and unstarted.
The renewed outage no longer constitutes a current blocking condition; the
goal remains active and unmet.
