# Frozen corpus ready and full100 preparation started

**Status:** STOPPED — incorrect 100-history scope; replaced by 100 questions on one history.
**Date:** 2026-09-15, updated 21:50 UTC.
**Scope:** Existing frozen candidate and corpus, execution evidence, and current handoff. No serving-method change.
**Depends on:** [Research Log 216](216%20-%202026-09-15%20-%20Resume%20interrupted%20frozen%20corpus%20evaluation.md), [frozen runner in Research Log 214](214%20-%202026-09-15%20-%20Complete%20exchanges%20and%20bind%20frozen%20full100%20evaluation.md), and [independent audit in Research Log 215](215%20-%202026-09-15%20-%20Independent%20full100%20result%20audit%20prepared.md).

> The user clarified **100 questions, not 100 histories**. Both processes below
> were stopped at 21:55:45 UTC, before any answer release. Do not resume this
> controller. Follow [Research Log 218](218%20-%202026-09-15%20-%20Correct%20scope%20to%20100%20questions%20on%20one%20ingested%20history.md)
> for the corrected run. The remainder of this log is historical evidence.

All missing parent hierarchies and summary vectors are complete. Both children
exited 0, and the existing controller automatically started full100 preparation.
This completes corpus materialization; population admission, actual answers,
judging, and the independent result audit remain required.

## Completed evidence

| Artifact | Verified result |
| --- | --- |
| Combined parent result | `eval_results/native-spine-frozen-corpus-20260914-r1/remaining-parents/result.json` |
| Parent result SHA-256 | `385939f425266aed02fe650ad369f25a80dfd8ef08700c144777adfa4c9b3cd4` |
| Parent bodies / templates | `31,166 / 31,166` |
| Reused / newly completed parent bodies | `13,768 / 17,398` |
| Parent completion | `2026-09-15T21:16:39.106911+00:00`, exit `0` |
| Vector result | `eval_results/native-spine-frozen-corpus-20260914-r1/vectors/result.json` |
| Vector result SHA-256 | `462a3e03986817411a82e5437b6d884bf169a7098f19a2c615ae1c27e1409eff` |
| Vector rows / dimensions / checkpoints | `323,124 / 1,024 / 2,525` |
| New / reused embedding rows | `180,278 / 142,846` |
| Vector completion | `2026-09-15T21:46:40.987632+00:00`, exit `0` |
| Completed vector log SHA-256 | `a1f5ac71a9eb8c6edf6a1517a5d223f2be8e11e67794213373b45dac0437fdde` |

Both result checksums were independently compared with their sidecars. Parent
completion preserves original atomic addresses and has `raw_inputs_to_qwen=false`.
The vector compiler's completed reader validates every matrix's hash, float32
shape, finite entries, and unit normalization. Its result has
`raw_inputs_to_models=false` and zero remote calls. Full100 target flags remain
false; these artifacts do not contain an answer-accuracy measurement.

The r3 parent continuation used 191 local jobs in 54 generation batches, bringing
the cumulative total to 1,714 within the original 4,096 allowance. It reused all
61 pre-interruption checkpoints and saved seven more. Research Log 216 records
the exact reconstruction and missing-checksum repair, including its four passing
checks. No frozen compiler, candidate, evaluator, or controller code was changed.

## Current execution

| Item | Binding |
| --- | --- |
| Controller root | `eval_results/native-spine-frozen-stages-20260915-r3` |
| Controller preflight | `29daf7ff0de41794e305dd9d0e915093168c94193a3549ba317cc93d68246c28` |
| Live exec session | `82986` |
| Controller | PID `19760`, creation time `1789504269.6213071` |
| Preparation worker | PID `47904`, creation time `1789508801.2392695` |
| Preparation started | `2026-09-15T21:46:41.244900+00:00` |
| Current log | `eval_results/native-spine-frozen-stages-20260915-r3/03-evaluation-prepare.log` |
| Full100 root | `eval_results/native-spine-frozen-full100-20260915-r1` |

The preparation worker first admits the complete 100-namespace population with
at least one million eligible body tokens per question, then constructs the two
memory packets and identical-prompt candidate API controls. The gold gate stays
closed until all 300 fresh answer streams are saved. The next controller stage
executes those streams and 200 logical judgments under the already frozen plan.

At 21:50 UTC, the preparation log reported seven admitted full1M namespaces,
and session 82986 was confirmed live with the same preparation worker. These
checks recount actual raw turn tokens through each question's date and verify
complete namespace materialization plus vector coverage. The population
admission artifact is intentionally absent until all 100 histories pass; a
partial progress log is not a complete admission receipt. No answer call has
started at this observation.

The target remains at least 95 correct candidate answers out of 100 with a warm
median below five seconds, normal stream termination, and matched direct API
timing reported alongside it. Tail latency and the fraction below five seconds
must be reported; the old 1.10 ratio gate does not apply. The development 8/8
result does not satisfy this benchmark target or establish generalization.

Inspect the active processes and log from the worktree:

```powershell
Get-Process -Id 19760,47904 | Select-Object Id,StartTime,CPU
Get-Content -LiteralPath 'eval_results/native-spine-frozen-stages-20260915-r3/03-evaluation-prepare.log' -Tail 10
```

Follow session 82986 while that same controller is live. Match PID and creation
time to the saved receipts; do not restart because a poll timed out. The
controller proceeds serially and records a stage exit before the next child.
After the joint report exists, run the independent audit from Research Log 215:

```powershell
.\.pixi\envs\dev\python.exe -X utf8 -m tools.audit_frozen_native_spine_full100 --root eval_results/native-spine-frozen-full100-20260915-r1
```

No actual full100 answer or accuracy result exists at this update. Keep the goal
active until the complete report and independent audit establish every target.
