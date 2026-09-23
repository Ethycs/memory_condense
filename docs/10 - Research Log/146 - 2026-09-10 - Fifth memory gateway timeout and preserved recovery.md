# Fifth memory gateway timeout and preserved recovery

**Date:** 2026-09-10  
**Status:** original jobs terminal; gateway probes failed; recovery staged without calls  
**Predecessor:** [145 - Qualified reader and complete memory development40 preflight](145%20-%202026-09-10%20-%20Qualified%20reader%20and%20complete%20memory%20development40%20preflight.md)

The goal turn made progress: the fourth complete memory's indexes finished,
the v3 reader and full100 report passed 82 focused checks, and 240 matched
requests were prepared for all 40 questions in the four complete memories.
The subsequent gateway failure stopped the scheduled comparison before any
answer request. The measured result remains 26/30 from Log 144; the full100
joint accuracy/latency target remains unverified.

## Confirmed terminal failure

Raw ingest session **38496** exited with code **1**, ending in
`openai.APITimeoutError` caused by `httpx.ReadTimeout`. The actual process,
PID **2856** with creation time **1789027208.5106108**, was subsequently absent.
The agent did not terminate it. This was confirmed through both the process
state and the original execution handle, not inferred from a polling timeout.

Comparison runner **73512** also exited with code **1**, reporting that its
ingest dependency exited unsuccessfully. It made **zero answer or judge
requests**. No timed measurements were collected during the failing ingest.
The runner and raw ingest are both terminal; do not poll or restart them as
though they were still waiting.

At terminal inventory:

| Raw request state | Count |
| --- | ---: |
| Completed responses and validations | 99 |
| Reserved without an authenticated response | 5 |
| Unstarted | 733 |
| Full namespace population | 837 |

There are 104 original request reservations. All remain at their original
paths. The five unacknowledged outcomes are unknown; they are counted as
physical attempts and are not relabeled as successful summaries. The original
full `atoms-prefix-0837.json` aggregate is absent.

The terminal observation is
`eval_results/full1m-spine-reader-v3-development40-20260910-r1/offset040-terminal-timeout.json`,
SHA `2d850af7d970bd4280dd4912afaeac6572ea5028d5e01a0b79c8dabb0cdab5b4`.
It also verifies that no reader answer reservations exist. A preceding attempt
to inventory a supposedly live process failed with `psutil.NoSuchProcess` and
wrote no live-state artifact; the terminal handle was then polled successfully.

## Model readiness

The bounded synthetic readiness protocol ran once under
`eval_results/spine-gateway-readiness-offset040-20260910-r1`. It used one
summary-only Qwen probe and one minimal Terra probe, each with a 30-second
timeout and zero retries. Neither probe contained benchmark or raw corpus
content. Both returned **HTTP 500 / InternalServerError** with no saved
completion. The two reservations are preserved; they must not be resent under
that root.

Preflight SHA:
`0bfaad0e01542cb37ce9b176981cce34271f680b6e55b00d4fa9ef960791ff25`.
Report SHA:
`84de324a19820ea63271482df99b48e4ab956226fa66961016bd93dab32f70c1`.
Qwen observation time: **2026-09-10 08:22:21 UTC**; Terra: **08:22:48 UTC**.
Readiness session **25139** exited normally after recording both failures.
The root's successful process exit does not mean model readiness passed.

## Explicit recovery stage

`tools/stage_spine_transport_recovery_v2.py` adds a complete-namespace staging
path for this terminal failure. It checks that the original process is absent,
reconstructs the entire prepared request population, and compares all journal
states against the recorded terminal observation. Changed source states require
a revised inventory. It preserves completed request/response files byte for
byte and leaves unresolved reservations at their original paths.

Root: `eval_results/spine-transport-recovery-offset040-20260910-r1`.
The stage contains all 837 prepared requests, 99 copied completed checkpoints,
and **738 explicitly declared future calls**: 733 first attempts plus at most
one additional attempt for each of the five unresolved requests. No completed
request is eligible for regeneration. Original and successor attempts must be
reported together; a complete recovery would have 842 raw physical attempts
for 837 logical requests, including the five original unknown outcomes.

- Stage preflight: `71010b979457a192ae74cf3af7669b343c0cd4a9d919bb3c10f008f120519c39`.
- Stage result: `154ec5a5b542cc82d558aa19ef84fb4a71cb36d76fc6ad6956d88cb2e91dcc07`.
- Verification: `6ce8ce473a4000e78be38185ef304c796dea2d1fb5732a0963c220bc08151d99`.

Staging replayed all 99 completed responses with zero calls. Verification then
checked all **198 copied journal files** against both their original and
successor bytes, confirmed that all 738 new execution slots are unstarted,
and verified that the legacy full100 transport verifier rejects this new
stage. Source atoms from this recovery therefore cannot silently inherit a
certificate that omits its five additional attempts. Staging session **92412**
finished successfully. The v2 staging implementation is now bound by its real
preflight and must remain reproducible.

## Next actions

**Continuation:** [Log 147](147%20-%202026-09-10%20-%20Preserved%20recovery%20execution%20and%20resumed%20reader%20comparison.md)
records the completed recovery executor, four-memory version-7 admission
replay, successful renewed readiness and resumed prepared reader comparison.
The following was the checkpoint before that continuation.

No known provider, indexing, preparation or comparison job remains live at this
checkpoint. Do not restart the two failed original processes or clear any
reservation. The four complete memories and all 240 reader requests remain
ready and unchanged.

The original comparison runner specifically requires the original fifth-memory
aggregate, so it cannot simply resume after staging recovery into a new root.
A successor may bind the confirmed terminal failure and fresh successful model
readiness before evaluating the existing four complete memories. That changes
the scheduling dependency, not the frozen reader/evidence comparison. Preserve
the original stopped runner and its plan.

Complete the bounded recovery executor and transport-admission verifier for
the new stage, with tests for retained successes, original reservation
accounting and stopped-run handling. The old recovery executor is specifically
bound to offsets 10/20 and must not be used for offset 40 by relabeling inputs.
Then obtain a fresh, successful readiness observation before any new model
calls. Keep bulk ingest and all timed evaluation work separate.

This is the first turn observing this renewed gateway failure. The goal is
still active; no completion or blocked status was set. There is further local
recovery work available while external inference is unavailable. Offsets 50
through 90 remain prepared and unstarted, and confirmation200 remains unopened.
