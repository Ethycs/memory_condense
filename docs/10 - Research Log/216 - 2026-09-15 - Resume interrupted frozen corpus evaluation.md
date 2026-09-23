# Resume interrupted frozen corpus evaluation

**Status:** RECOVERY COMPLETE — parent and vector stages succeeded; full100 preparation continues in Research Log 217.
**Date:** 2026-09-15, updated 21:47 UTC.
**Scope:** Uncommitted continuation controller, tests, and execution handoff.
**Depends on:** [Research Log 214](214%20-%202026-09-15%20-%20Complete%20exchanges%20and%20bind%20frozen%20full100%20evaluation.md) and [Research Log 215](215%20-%202026-09-15%20-%20Independent%20full100%20result%20audit%20prepared.md).

> **Current execution:** Both materialization stages exited 0. Follow
> [Research Log 217](217%20-%202026-09-15%20-%20Frozen%20corpus%20ready%20and%20full100%20preparation%20started.md)
> for the completed vector binding and active full100 preparation worker.
> The process tables below are historical snapshots of the recovery.
> The subsequent 100-history preparation was stopped after the user corrected
> the scope. [Research Log 218](218%20-%202026-09-15%20-%20Correct%20scope%20to%20100%20questions%20on%20one%20ingested%20history.md)
> is the active 100-question, one-history handoff. Do not resume the old controller.

## Current continuation after exact checksum recovery

The r2 controller (session 55186) exited 1 at 20:26 UTC. It successfully
revalidated all 61 saved checkpoints, then encountered one body JSON without
its checksum sidecar in the first incomplete batch. The file was 68,077 bytes
and last written at 11:29:19 UTC, during the original interrupted execution.
The hierarchy directory contained 15,778 body files and 15,777 sidecars. No
additional model job was issued by r2. Its recorded failure is distinct from
the unknown cause of the original interruption.

[`repair_native_spine_parent_seal.py`](../../tools/repair_native_spine_parent_seal.py)
reconstructed that one hierarchy in a separate audit directory using the
unchanged compiler, authenticated saved summaries, and cached attention. The
reconstruction has no generation capability and rejects any missing summary.
Only after a byte-for-byte match did it exclusively create the absent checksum.
The original body bytes and modification time were preserved. A changed body
or any existing checksum is rejected, never overwritten. Four focused checks
passed in 1.91 seconds in
[`test_native_spine_parent_seal_repair.py`](../../tests/test_native_spine_parent_seal_repair.py).

| Repair evidence | Binding |
| --- | --- |
| Repair root | `eval_results/native-spine-parent-seal-repair-20260915-r1` |
| Repair result digest | `91df3d5b77fc99701c70174337265dfdc16c18c12886cccf66097ad9320004c4` |
| Restored body name | `f29d595cb55e9b50ce71398010c3ecb931b0fd6440d0e63d1fd9c2231b3f1f48.json` |
| Exact reconstructed/original artifact digest | `f03417a3a665ddae0db452f3f4ff0bd5c46bc56977012c03caaf8a18aba69d7e` |
| New model calls / body files replaced | `0 / 0` |

The repair command completed successfully:

```powershell
.\.pixi\envs\dev\python.exe -X utf8 -m tools.repair_native_spine_parent_seal --root eval_results/native-spine-parent-seal-repair-20260915-r1 --failed-controller eval_results/native-spine-frozen-stages-20260915-r2
```

Do not rerun that command against its existing root. Both r2 processes were
absent before repair and restart. The r3 preparation reauthenticated the same
15,616 saved bodies and 1,523 executed jobs. Its preflight digest is identical
to r2 because the frozen plan, checkpoint prefix, and generation transactions
are unchanged; the restored file belongs to the incomplete batch.

| Current execution | Binding |
| --- | --- |
| Controller root | `eval_results/native-spine-frozen-stages-20260915-r3` |
| Exec session | `82986` |
| Controller process | PID `19760`, creation time `1789504269.6213071` |
| Completed parent process | PID `56464`, creation time `1789504282.1710713`; exited 0 |
| Parent log | `eval_results/native-spine-frozen-stages-20260915-r3/01-parent-run.log` |
| Active vector process | PID `48544`, creation time `1789506999.808921` |
| Active vector log | `eval_results/native-spine-frozen-stages-20260915-r3/02-vectors.log` |
| Vector started | `2026-09-15T21:16:39.816580+00:00` |

At the 20:37 UTC check, r3 had revalidated all 61 completed batches and passed
the repaired artifact. The first incomplete batch assembled 230 bodies from
cached summaries before requesting its 26 missing merge jobs. Qwen verified
all five checkpoint shards, loaded with 4.508 GiB allocated, and completed its
first four new summary jobs in one 25.329-second generation batch. The accepted
merge cache reached 4,431 entries. This confirms new generation has resumed;
the complete-batch checkpoint total was still 15,616 at that observation.

At 20:40 UTC, the normal compiler sealed batch `0061.json`: 256 bodies, including
the recovered artifact with its exact digest above. Independent checksum reading
matched `d0962f4156e6661390234afbcc2d4d76c47f78b78538a4ef3be6a386e7e78304`,
and the checkpoint retained parent preflight
`bf0f0ef22e39ed4519f60bd1713351fe3ad27521db3a81aa7b04a420a6964b9f`.
There are now 15,872 completed missing parent bodies and 1,526 remaining.
R3 used 27 new summary jobs in eight generation batches to reach that checkpoint
(1,550 jobs cumulatively with the original run). The next group assembled 232
bodies from cached summaries and identified 24 pending merges. The same live
controller and child remain active; vectors and full100 answers have not started.

At 21:09 UTC, all 17,398 missing parent bodies were saved in 68 complete
checkpoints. R3 completed 191 new local summary jobs in 54 generation batches;
the cumulative job total is 1,714, within the original 4,096 allowance. The final
246-body checkpoint `0067.json` independently matched digest
`a1aedc5644fef9b7f1aa3be4a60a81277baed8dee3a69a3be1363727b622a589`
and retained the same parent preflight. There were no pending merge jobs.
The parent worker was still live, validating all 31,166 combined templates;
`remaining-parents/result.json` did not yet exist. Generation completion alone
does not establish combined cache admission or full100 accuracy.

At 21:16:39 UTC, the parent stage exited 0 and published the combined result:

| Combined parent evidence | Verified value |
| --- | --- |
| Result | `eval_results/native-spine-frozen-corpus-20260914-r1/remaining-parents/result.json` |
| Result digest, independently checked against sidecar | `385939f425266aed02fe650ad369f25a80dfd8ef08700c144777adfa4c9b3cd4` |
| Body count / template count | `31,166 / 31,166` |
| Reused / newly completed parent bodies | `13,768 / 17,398` |
| Complete native hierarchies / complete source compilation | `true / true` |
| Original atomic addresses preserved / raw inputs to Qwen | `true / false` |
| Full population admitted / full100 target passed | `false / false` |
| Completed parent log digest | `817140aa06e6f6a20b4fe5421ae9c22590c812360a1edf4dc45b31680cac78d0` |

The controller then started the vector child listed above, after Qwen exited.
The vector stage reuses authenticated embeddings and computes only missing
atomic-summary vectors. Its prepared population is 323,124 unique summaries.
At 21:19 UTC, its log reported 9,088 completed vector rows, and the process
matched the saved PID and creation time. Session 82986 remained live in the
vector stage. Vector completion is still pending.
The full100 preparation stage still must admit all 100 date-eligible namespaces
and bind their packets; parent-cache completion is not an accuracy result.

The frozen candidate, compiler, evaluator, and controller implementations were
not edited. After parents, this controller still runs vectors, full100
preparation, and answers/judgments serially. Independently audit the actual
joint report afterward using Research Log 215. No full100 answer result exists
at this update.

## Original interruption and first continuation

The original controller and parent worker were absent at the 20:10 UTC check,
and exec session 54878 no longer existed. The original parent log ended after
saving 61 complete checkpoints: 15,616 of 17,398 missing parent bodies. There
was no recorded parent exit, so the interruption cause and exit code are
unknown. This was an interrupted process, not a completed corpus or evaluation.

`tools/resume_frozen_native_spine_stages.py` prepares a separate controller root
and authenticates the original frozen plan, candidate, worker receipts, parent
preparation, checkpoint prefix, and completed generation journals. It refuses
to run while either original process is live or if a later stage already has
output. It also rejects an executed request without a saved response. The
existing OS lifecycle lock is reused without deleting its file.

The read-only preparation confirmed:

| Evidence | Verified value |
| --- | --- |
| Completed missing parent bodies | 15,616 of 17,398 |
| Previously reserved local jobs | 1,523 |
| Remaining original local-job allowance | 2,573 of the original total 4,096 |
| Authenticated accepted new summary keys | 1,465 |
| Authenticated transaction digest | `98951e772f610cfb4387ceeec981e1233203c26a862e39a07b93d83858781895` |
| Continuation preflight digest | `29daf7ff0de41794e305dd9d0e915093168c94193a3549ba317cc93d68246c28` |

The continuation uses the original stage commands in order: remaining parents,
vectors, full population admission and evaluation preparation, then the frozen
100-question answer/judge run. The parent compiler revalidates saved batches
and skips their compilation. Its total job limit remains cumulative across
both invocations. The frozen candidate, summary-only Qwen inputs, exact raw
hydration, answer population, and timing protocol retain their original bindings.

Nine focused checks passed in 19.13 seconds: seven continuation checks plus the
existing completed-batch reuse and interrupted-generation refusal checks. The
checks cover live-worker refusal, changed or missing checkpoint populations,
serial stage order, stopping after failure, exclusive answer release, changed
source state, and preserving a failed target even when the workflow completes.
These are recovery checks, not answer-accuracy evidence.

The new controller started successfully and launched the parent worker. Its
first logged completed batch was reused with zero new local jobs.

| Historical r2 execution — exited 1 | Binding |
| --- | --- |
| Controller root | `eval_results/native-spine-frozen-stages-20260915-r2` |
| Exec session | `55186` |
| Controller process | PID `58280`, creation time `1789503465.5799649` |
| Parent process | PID `46320`, creation time `1789503476.7037551` |
| Parent log | `eval_results/native-spine-frozen-stages-20260915-r2/01-parent-run.log` |
| Original parent cache | `eval_results/native-spine-frozen-corpus-20260914-r1/remaining-parents` |
| Expected evaluation root | `eval_results/native-spine-frozen-full100-20260915-r1` |

The complete accuracy/latency report and the independent result audit are still
required. The full100 target remains unverified. After the joint report exists,
run the independent audit command in Research Log 215.

To inspect the active continuation from this worktree:

```powershell
Get-Process -Id 19760,47904 | Select-Object Id,StartTime,CPU
Get-Content -LiteralPath 'eval_results/native-spine-frozen-stages-20260915-r3/03-evaluation-prepare.log' -Tail 10
```

Match the processes to the saved worker receipts before treating them as live.
Continue following session 82986 while that same controller is active. If it
becomes terminal, inspect its recorded stage exit and log before any further
continuation. A wait timeout alone is not evidence of termination.
