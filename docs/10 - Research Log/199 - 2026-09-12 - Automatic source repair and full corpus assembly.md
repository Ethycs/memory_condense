# Automatic source repair and full corpus assembly

**Status:** A real repair coordinator is running alongside original ingestion.
It will assemble the complete corpus after ingestion and repairs finish. The
joint 95/100 accuracy and API-latency target remains unverified.

The preceding alias-answer turn did not advance the benchmark. This continuation
verified the actual running processes, observed the completed R4 vector job, and
replaced repeated manual repair-cohort preparation with a bounded coordinator.

## Source completion coordinator

`tools/complete_native_spine_ingestion.py` authenticates completed repair journals
before excluding them from its population. It observes the existing source
producer's exact PID and creation time and selects only published, rejected
original validations. It never dispatches an original source request. Ordinary
attributable failures use the existing exact-section repair tool; malformed JSON
uses the separate JSON recovery tool. Other structural failures stop execution.

Each new cohort records its selected original validations. Each subdivision stage
has a distinct root, preflight and call allowance. Accepted original summaries
and accepted repair sections remain unchanged. An uncertain provider request,
exhausted subdivision allowance or stopped source producer ends the coordinator;
there is no implicit restart or transport retry.

The coordinator accounts for the two-step publication of validation JSON and its
digest sidecar. While the source process is live, a file awaiting its sidecar is
revisited on the next poll. After the process exits, validation is strict. A stale
completion receipt cannot override a live process with the recorded identity.

After the original producer exits with a matching successful terminal receipt,
every prepared ordinal must belong to an original validated response or an
explicit transport recovery, and every rejected original must have a repair.
Only then does the existing JSON-aware assembler run with `allow_partial=False`.
It replays the actual source/repair records and validates exact full-body raw
coverage. The new store must preserve every R6 body and section and contain all
31,166 prepared bodies. This work does not read evaluation questions or gold.

| Item | Value |
| --- | --- |
| Root | `eval_results/native-spine-source-completion-20260912-r1` |
| Policy SHA | `ffee935741d93ea3f130922052b97a6462cc6d7af7c18cd552a2e2d02daa1fed` |
| Started SHA | `9baeab0d3c70d1b699d6673bf4b059d605fea51b7c499adb159c33db830595ea` |
| PID / creation time | `54564` / `1789235160.7269917` |
| Session | `32013` |
| Existing repaired originals | 286, including final R17 and JSON batch 4541 |
| Existing transport recoveries | 6 |
| Maximum additional provider calls | 2,048 across all new cohorts and stages |
| Cohort size / subdivision stages | At most 32 original batches / 6 stages |
| Collection threshold | 16 rejected batches, or 15 minutes, or source completion |
| Poll / total time allowance | 30 seconds / 24 hours |
| Final body-store location | `complete-body-store/` beneath the coordinator root |

Preparation session 16956 exited zero after replaying all seed lineages with no
new calls. Its policy binds 24 implementation files. The configuration is
`.tmp/native-spine-source-completion-20260912-r1.json`; the policy contains the
resolved input bindings and limits used by execution.

The first live scan found 6,589 accepted and 311 invalid original validations.
Its cohort selects 25 previously unrepaired batches: 24 attributable section
failures and malformed JSON batch 6490. Selection SHA:
`c2b397f1637c8d46163caea6fa1a90c34b39adda84219c2d1e9756e89038fabd`.
The first direct stage preserves 520 valid original summaries and prepares 103
replacement sections in at most 13 Terra calls. Its preflight SHA is
`2791870ec3152460ce4305344e610fb1c7a709ef600a942ec11da69fa76a94aa`.
The first two real calls completed and admitted all 16 returned sections.
The cohort and final corpus are not yet complete.

Do not launch a manual repair for any coordinator-owned original or restart the
coordinator while this exact process is live. `cohorts/*/finished.json` records
the sole final repair roots for each completed cohort. If it fails, inspect its
failure and current-stage journals before choosing any recovery. The explicit
stop flag is `stop-after-current-cohort.flag` beneath the coordinator root.

## Verification

Fourteen new checks pass. They exercise actual repair journals, both failure
classifications, unchanged originals, selective subdivision, call and stage
limits, uncertain transport calls, duplicate and in-flight validation files,
complete request accounting, and process identity versus terminal receipts. A
complete fixture executes the coordinator through both repair types and actual
full-store assembly, then confirms a second invocation cannot send more calls.
Fixture providers make no remote calls and establish no model accuracy.

The initial run had 12 passes and one test-fixture failure: Windows text output
translated the synthetic digest sidecar to CRLF. Writing its required LF bytes
fixed the fixture; that test passed separately in 1.68 seconds. The additional
full coordinator test passed in 3.14 seconds. No production repair behavior was
changed to satisfy that fixture.

All 113 files bound by the current full100 readiness result remain unchanged.
The active exchange, attention, parent and vector controllers' bound files also
remain unchanged. No actual native full100 answers or judgments have been sent.

## Other compilation progress

Vector R4 finished all 118,693 unique summaries: 74,059 reused and 44,634 newly
embedded. Session 64473 exited zero and PID 22840 is gone. Result SHA:
`72fbd0ad201f85bf134641a7644466e81da3cce0f795cfcc09b2e06b2749ba56`.
Terminal handoff SHA:
`c2732c443a953e3e0a1dd666817a153b9196a724f417fca7f70e80eb4d3c7fb3`.

That completion released the existing R4 exchange controller (PID 49268/session
55723). It replayed the 13,316 completed exchange sets and then loaded the actual
local Qwen checkpoint. The observed first 68 new jobs advanced the accepted
merge cache from 317 to 359 keys; the expanded exchange population remains in
progress. Attention R5 and the expanding parents retain their existing queue.

Vector R5 has now prepared 139,694 unique R6 summary inputs, reusing completed R4
vectors without loading BGE. It waits for the expanded parent process to exit
before encoding. Preflight SHA:
`88c0f24ef662833aaffecdda95d7e8f5dba7ed436e4ed62c984fe882798408ac`.
Prepared receipt SHA:
`19d85c9f8449d701d39e79b8b4ab011f3100a7dc95191b2375524f1bfd9ba152`.

These queued GPU jobs cover the R6 snapshot, not the eventual full corpus. After
the coordinator publishes its complete store, the reusable exchange, attention,
parent and vector producers must cover that full store before the unchanged
native full100 evaluator can run. The required result is still at least 95/100
and all eight matched latency ratios at most 1.10 in the same fresh evaluation.
