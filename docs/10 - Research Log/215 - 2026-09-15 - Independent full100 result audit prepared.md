# Independent full100 result audit prepared

**Status:** Audit code prepared and checked; the real evaluation is still pending.

**Current execution handoff:** The original worker stopped after saving 15,616
missing parent bodies. Follow [Research Log 216](216%20-%202026-09-15%20-%20Resume%20interrupted%20frozen%20corpus%20evaluation.md)
for the authenticated continuation and current session. The audit procedure below
still applies after the full100 report exists.

The preceding goal turn was a verified wait with completed work: four parent
checkpoints reached 1,024 missing bodies. This turn prepared a separate final
result audit while the same resident parent worker advanced to 1,536 checkpointed
bodies. The frozen candidate and active stage controller were unchanged.

`tools/audit_frozen_native_spine_full100.py` will audit the completed result in
`eval_results/native-spine-frozen-full100-20260915-r1/`. It requires the saved
joint report and all 300 bound answers before opening references. It verifies
the 200 logical judge inputs against the measured responses and locked reference
answers, authenticates the saved judge journals without a provider, and checks
that the reported verdicts are the ones those journals contain.

Accuracy, median, nearest-rank p95, mean, matched API ratios and the number of
candidate responses below five seconds are recomputed independently of the
evaluation runner's statistics function. The audit checks the declared 95/100
and warm-median-under-five-seconds decision while retaining the measured tail
and normal-completion status. It does not infer generalization beyond this
historically exposed benchmark population.

For both memory packets on every question, the audit resolves the selected
source occurrence in the bound original namespace and reads its body from the
original SQLite bank. It verifies body identity, original turn identity,
speaker, occurrence date, character coordinates and the exact served bytes.
It reconstructs the raw packet and reader messages and compares them with the
saved API messages. Raw evidence from a foreign or future occurrence is rejected.
The full population admission remains separately bound to the evaluation;
this excerpt audit does not substitute for the required complete 1M histories.

Nine focused checks pass. Eight passed in the initial run; the remaining test
was corrected to expect the artifact reader's actual `SealedArtifactError` for
a missing report, then passed separately. The correction changed only the test.
Coverage includes known 95/100 versus 94/100 outcomes, the five-second threshold,
visible p95 above five seconds, changed response bindings, incomplete answer
populations, exact Unicode raw text, forged text, changed original bodies,
foreign occurrences, future evidence, altered routing and packet budgets, and
the reference-opening gate. These checks establish audit behavior, not real
benchmark accuracy or latency.

After the controller finishes the actual evaluation, run:

```powershell
.\.pixi\envs\dev\python.exe -X utf8 -m tools.audit_frozen_native_spine_full100 --root eval_results/native-spine-frozen-full100-20260915-r1
```

The audit is an explicit subsequent step; it was not inserted into the already
running, hash-bound controller. It writes `independent-audit.json` and makes no
model calls. A failed accuracy or latency threshold remains a failed target
even if the result audit itself passes.

At the subsequent 08:36 UTC observation, controller session 54878 remains live
with parent child PID 55800. Sixteen 256-body checkpoints are complete: 4,096 of
the 17,398 missing parent bodies, with 396 local jobs and 382 accepted new
summaries. The latest checkpoint contains 256 bodies, retains the parent
preflight binding, and its actual SHA matches its saved digest:
`736bcf360f0ded3e71bee74dfd3e4c0666f33dc913c6f2a9d3477383909399e1`.
Process CPU activity advanced and observed working memory was about 3.7 GB.
Continue following that same session and
`eval_results/native-spine-frozen-stages-20260915-r1/01-parent-run.log`. The next
stages remain parent completion, missing vectors, complete population admission,
fresh 100-question evaluation, and then this independent result audit. The goal
remains active and no native full100 score has been established.

At 09:37 UTC, the same live controller and parent child have reached 32 complete
checkpoints: 8,192 of the 17,398 missing parent bodies. At that checkpoint the
worker had completed 793 local jobs in 225 generation batches, adding 767
accepted summaries to the 2,962-key seed. The saved `batches/0031.json` contains
256 completed body bindings; its parent preflight binding matches, and its
actual SHA matches the sidecar:
`20a9e5b6c3fac7994f41fc9e5dc529a84ae7decc07116b1f20545fe63261ff54`.
The parent process retained its original creation time, had accumulated
6,497.98 CPU seconds, and used 3.522 GiB of resident system memory. No stage
failure was reported. These are preparation measurements; the full100 answer
run and independent result audit remain pending.

The README was reconciled to identify the first pilot as historical and point
its next action to the active frozen continuation. Those documentation changes
are uncommitted. `git diff --check -- docs/README.md` passed.

To recheck the current worker from this worktree:

```powershell
Get-Process -Id 55800,73612 | Select-Object Id,StartTime,CPU
Get-Content -LiteralPath 'eval_results/native-spine-frozen-stages-20260915-r1/01-parent-run.log' -Tail 10
```

Match the child creation time to `2026-09-15T00:29:54.8013323-07:00` and the
controller to `2026-09-15T00:14:39.8472412-07:00`; PID reuse is not proof that the
original worker is live. Continue following session 54878 while that same worker
is live. Inspect the controller's recorded stage exit and log if it becomes
terminal, then resolve the actual failure before any continuation. Run the
independent audit above only after the full100 joint report exists.
