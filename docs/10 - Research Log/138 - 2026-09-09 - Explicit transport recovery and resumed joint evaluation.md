# Explicit transport recovery and resumed joint evaluation

**Date:** 2026-09-09 (local; execution observations are September 10 UTC)  
**Status:** recovery snapshot; completed comparisons and current continuation in Log 139  
**Predecessor:** [137 - Gateway timeouts and complete-memory recovery handoff](137%20-%202026-09-09%20-%20Gateway%20timeouts%20and%20complete-memory%20recovery%20handoff.md)

Both model routes passed fresh bounded readiness probes after the earlier
timeouts. Qwen returned in 1.511 seconds and Terra in 5.031 seconds. The probe
used the same synthetic inputs and two-call limit as before; it sent no corpus
raw text or benchmark question. The complete 1M-token overflow comparison is
now running in session **67654**, with all bulk recovery and compilation
deferred until its answer timing finishes. No new answer accuracy is claimed.

Readiness root: `eval_results/spine-gateway-readiness-20260910-r2`.
Preflight SHA: `0bfaad0e01542cb37ce9b176981cce34271f680b6e55b00d4fa9ef960791ff25`.
Report SHA: `68f07359bfd216c7c7f3c8503661123ad818332c346b27fdeddb5b45c2667b09`.
Both responses are authenticated in their journals. This resolves the observed
inference-availability blocker; it does not establish the earlier requests'
gateway-side outcomes.

## Later status

The answer comparison described above has finished, followed by a second
complete-memory comparison. Their overflow candidate scored 18/20 in total,
versus 16/20 for fresh controls. Both reports replay with zero judge calls.
Offset 10 is admitted, indexed, and version-2 verified; offset 20 raw ingest
also completed. The nine over-budget summaries in offset 20 require a new
multi-batch admission path. The running-session references and next steps
below describe the earlier recovery snapshot. Current evidence and next work
are in [139 - Two complete memory overflow comparisons](139%20-%202026-09-09%20-%20Two%20complete%20memory%20overflow%20comparisons.md).

## Preserve completed work and count every attempted request

`tools/stage_spine_transport_recovery.py` replayed **1,385 completed raw
requests** without provider calls: all 801 from offset 10 and the 584 completed
requests from offset 20. Their request and response bytes were copied into a
separate execution root. All original failed reservations remain in place.
No completed response is eligible for replacement, and the original runtime's
refusal to retry an unacknowledged request is unchanged.

Recovery plan: `eval_results/full100-spine-corpus-20260909-r1/transport-recovery-plan-20260910-r1.json`,
SHA `237a8c130e2417bc569bf8459203d60c42d74cc57f1d08f69c40b76afc0d1aad`.
It permits at most **234** new calls: 228 first attempts for unstarted raw
requests, five explicit additional raw attempts, and one explicit additional
summary-compaction attempt. This is a ceiling before any operator-recovered
responses reduce the remaining work. There are no automatic SDK retries.

Stage root: `eval_results/spine-transport-recovery-20260910-r1`.
Stage SHA: `446ab81e6e3fd8fbbf3f396285817a08e7e58251d1a7862ff5d14d8e596519ed`.
The staged corpus is under `corpus`; the staged Qwen compaction is under
`summary-repair-offset010`. Original request protocols and corpus bindings
reproduce exactly. The frozen plan and stage recorded that execution was not
implemented at their creation; the subsequently added wrapper below supplies
that implementation without rewriting these historical preparation records.

`tools/execute_spine_transport_recovery.py` exposes separate `compact` and
`raw` phases. Before any new call it verifies retained successes, the original
failure inventory, the remaining call count, and fresh successful readiness
responses for both routes. Readiness must be observed within five minutes and
replay from authenticated completion journals; a status flag or HTTP liveness
response is insufficient. Any unresolved request in the successor itself is
refused, so this wrapper cannot silently add a third attempt.

`tools/probe_spine_gateway_readiness.py` prepares and executes the fixed
synthetic two-model probe. It has a 30-second timeout, 64-token output cap,
zero retries, and records HTTP status codes when available. It refuses to
resend a readiness request with an existing reservation.

## Admission version 2

The old admission certificate allowed one compaction call per namespace. It
cannot truthfully account for the failed original call plus its successor.
`tools/verify_spine_admission_method_v2.py` therefore uses a new certificate
format and method identity. It preserves the original content-admission rules,
permits at most one additional transport attempt and one successful compaction
batch, and records the earlier unknown attempt conservatively. The original
certificate and source atoms remain unchanged.

`tools/spine_transport_lineage.py` verifies each retained success and explicit
reissue against the bound stage and original journals. It rejects changed raw
requests, replacement of a prior success, missing successor responses, altered
compaction protocols, and a changed original outcome inventory. Admission then
replays the complete current response population through the original source
and summary-budget checks. No original raw input enters a Qwen request.

On the complete first memory, version 2 reproduced all **5,556 atoms** using
850 raw completion hits and one compaction hit, with zero calls. Certificate:
`eval_results/full100-spine-corpus-20260909-r1/offset-000/conditional-method-v2-prefix-0850.json`,
SHA `8ccf6a5c77555d48c5db0a1bc329a3b9e9175c2e03d13b52d549b2720ee7c4fd`.
Method SHA: `613f02a94a834bf4ac7c0dc9112e0331314296d45bc5e817aa39a2551304995d`.
This memory needed no transport recovery. The staged memories cannot receive
their version-2 certificates until their missing inference results exist.

Use `tools/report_joint_source_spine_overflow_full100_v2.py` for the eventual
aggregation. It requires the version-2 admission proof on every memory and
retains the same 100-question, 1M-token, 95%-accuracy and matched-latency gate.
The answer harness, reader policy, routing, evidence budget, and frozen
50-call comparison are unchanged.

## Verification and next execution

The transport staging, lineage, readiness, admission replay, and original plus
versioned full100 gates pass **26 focused tests**. They cover authenticated
replay, retention of failed reservations, foreign evidence, resealed replacement
of prior successes, compaction attempt accounting, stale and failed readiness,
and refusal to pass from partial or mixed answer populations. A real-data
check also confirmed the recovery wrapper rejects the failed readiness report
before creating a new reservation or execution preflight.

Finish session 67654, then judge its complete answer population and replay the
judgments with zero calls. Its preflight remains
`a8be6df727e24fcebb5e5db329276634f1230b56980bec3749fef8796e4d6921`.
Afterward obtain fresh successful readiness if the prior observation is stale,
execute the staged compaction, admit and verify offset 10 with version 2, and
compile its attention-partitioned leaves and summary indexes. Resume offset
20's staged raw work within its 233-call ceiling. Complete all remaining
memories and score one unchanged method across the full100 population.
