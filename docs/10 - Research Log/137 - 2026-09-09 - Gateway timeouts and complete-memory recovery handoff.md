# Gateway timeouts and complete-memory recovery handoff

**Date:** 2026-09-09 (local; observations extend into September 10 UTC)  
**Status:** inference unavailable in bounded probes; complete saved work preserved  
**Predecessor:** [136 - Preserve selected user evidence before attached context](136%20-%202026-09-09%20-%20Preserve%20selected%20user%20evidence%20before%20attached%20context.md)

The next accuracy comparison has not started. Both outstanding gateway jobs
ended with ten-minute inference timeouts. Two later small readiness probes
also failed: Qwen timed out after 30 seconds and Terra returned a server error.
This points to a gateway or worker problem rather than the ingest payload size.
The last measured answer score remains the reader comparison's 9/10 on one
complete memory; the joint full100 target remains unmet.

**Later status:** fresh readiness passed on both routes, and the frozen answer
comparison resumed. [Research Log 138](138%20-%202026-09-09%20-%20Explicit%20transport%20recovery%20and%20resumed%20joint%20evaluation.md)
records the preserved recovery stage, versioned attempt accounting, and live
evaluation. The timeout observations below remain unchanged.

## Saved execution state

| Work | Authoritative state |
| --- | --- |
| Offset 10 raw ingest, session 85650 | Finished: 801 responses, 1,044,341 raw tokens, exit code 0 |
| Offset 10 full admission audit | Five over-budget summaries; no schema or attribution failures |
| Offset 10 Qwen compaction, session 43793 | Exit code 1, `APITimeoutError` caused by `ReadTimeout`; one reservation, no saved response |
| Offset 20 raw ingest, session 91664 | Exit code 1, same timeout types; 584 completed, five unresolved, 228 unstarted |
| Readiness probes, session 15042 | Finished; both probes failed, no completed inference responses |

All these process handles are terminal. No local provider or GPU compilation
job remains live. The gateway-side outcomes of unacknowledged requests are
unknown; local termination does not prove remote cancellation.

The sealed inventory authenticates the raw request and response journals:
`eval_results/full100-spine-corpus-20260909-r1/gateway-timeout-inventory-20260910-r1.json`,
SHA `2fabd4638a6ce1590fbfe114184bee7eb87c520a0f484b7b2303607ff5489af4`.
Offset 20's unresolved zero-based batch indices are **584–588**. Its completed
prefix audit remains at SHA
`8fc427b378a91306f359be9aa4afeba83458193572c79d4ef473ca09b3afbf1c`:
five over-budget summaries, no schema or attribution failures, zero audit calls.
Do not interpret missing response journals as proof that requests never ran.

Offset 10's complete audit and five-summary compaction preflight remain at
SHAs `5850c40483dc683b926cb91d37cfdf762080835d3443a26b7e4af221dd579bd3`
and `d130c629a127c9e1fcfdc15f997fb510b721751e7ccf2e84326d17a07a9999c0`.
No original request, response, or failed reservation was deleted or rewritten.

## Gateway diagnosis

DNS, TCP, and TLS connections succeeded. `/health/liveliness` returned HTTP
200 with the repository's Windows trust-store TLS configuration. A first probe
using the default HTTP client's different CA store failed; that result is not
evidence of a network outage. The gateway's live OpenAPI schema confirms that
`/health/backlog` reports in-flight HTTP requests and `/spend/logs/v2` provides
paginated request logs. The current credential received HTTP **403** from
backlog and from logs filtered to its own key hash and the relevant models.
Thus those endpoints cannot currently establish worker status or recover
attributable responses. No alternative credential or access path was used.

The separate readiness preflight allowed exactly two calls, zero retries,
30-second timeouts, and a 64-token output cap. Qwen received only a synthetic
summary; Terra received a short readiness instruction. Neither probe used
benchmark questions, gold answers, or corpus raw content. They did not retry
any original unresolved request.

Root: `eval_results/spine-gateway-readiness-20260910-r1`.
Preflight SHA: `0bfaad0e01542cb37ce9b176981cce34271f680b6e55b00d4fa9ef960791ff25`.
Report SHA: `3b15e28d583aa495e577d8039cc079fd2a04135ec798086efd9dfa1ab8ddf070`.
The Qwen observation records `APITimeoutError`; Terra records
`InternalServerError`. Error bodies were not retained, so a more specific
backend cause is unproven. The user has been asked to check the workers.

## Resume from preserved work

1. Establish inference readiness after the worker issue is resolved. A healthy
   gateway liveness endpoint alone does not establish model availability.
2. Recover attributable terminal responses from the operator if available.
   Otherwise prepare a bounded successor attempt that explicitly records the
   earlier unresolved attempt before reissuing any of those requests. Preserve
   original journals and account for attempts separately from accepted results.
3. Complete offset 10's summary compaction, source-bound admission, conditional
   method verification, attention-partitioned leaves, semantic index, and user
   summary addresses. Its complete raw input does not need to be resent.
4. Continue offset 20 from the 584 saved responses, resolve all five outstanding
   requests, and execute the 228 unstarted requests. Audit the complete memory
   before preparing its summary compactions. Do not omit unresolved fragments.
5. Run the already prepared 50-call overflow comparison when inference is
   available and bulk work is quiescent. Its preflight and dependencies remain
   unchanged; follow with judging and zero-call replay.
6. Continue the remaining seven prepared memories and apply the same full100
   accuracy and latency gate. No partial memory or ten-question result can
   substitute for that target.

This handoff changes operational state only. No new answer score, model
promotion, weakened admission rule, or target pass is claimed.
