# Full corpus compilation and joint evaluation handoff

**Status:** The complete compilation/evaluation handoff is running and waiting
for its six recorded predecessors. No full-corpus model stage or native full100
answer has been released. The joint target remains unverified.

The previous goal turn made concrete progress by starting automatic source repair
and complete-store assembly. This continuation verified those actual live jobs,
observed the first repaired cohort finish, and connected the eventual complete
store to the remaining GPU work and one fresh joint full100 evaluation.

## First automatic cohort completed

The source-completion worker remains PID 54564, creation time
`1789235160.7269917`, session 32013. Its first cohort repaired all 25 selected
original batches in 19 new Terra calls. The 24 attributable failures required
13 initial calls and two plus one calls for selective subdivision. The final
direct result preserves all 520 valid original summaries. Malformed JSON batch
6490 recovered all 24 original fragments in three separate calls.

| Artifact | SHA-256 |
| --- | --- |
| `cohorts/0000/finished.json` | `7ea68f2b39c53cf46a51610e874143566647b257a8b08345330e780a7950702d` |
| Final direct `cohorts/0000/direct_roots/stage-02/result.json` | `3e41b4a520fcd5bdb5f4d004ab2ce53362889983ba0d3e158dec8566cf1f2bdd` |
| JSON `cohorts/0000/json_repair_roots/stage-00/result.json` | `069aa62468975c8312421e183f48012003a1e65169f483c3a75df95d3ca6f22f` |

Paths in that table are beneath
`eval_results/native-spine-source-completion-20260912-r1`. Only the final roots
in the cohort receipt enter final assembly. There are now 311 completed repaired
originals, plus six separate transport recoveries. At the latest scan, original
validations numbered 6,737 accepted and 320 invalid; nine invalid originals
awaited the next repair cohort. Original statuses remain unchanged by repair.

Expanded exchange compilation also advanced: the first 128-job invocation now
completes 13,348 of 13,468 available body exchange sets, with 69,649 exchanges.
Its partial result SHA is
`3f571753f0ed7e0aec5e601b2b3a4c0a8ac92b7e01c8751934e8ad6a80deb822`.
The same live controller continued into its next invocation. Those inputs are
still the R6 partial-corpus snapshot.

## Full-corpus handoff

`tools/run_native_spine_full_corpus.py` waits for exact predecessor process
identities to exit and for their matching terminal artifacts. A live PID with
the recorded creation time takes precedence over a completion file. Missing,
failed or partial terminal output stops the handoff before downstream work.
No stage automatically retries or restarts an uncertain request.

| Item | Value |
| --- | --- |
| Output root | `eval_results/native-spine-full-corpus-pipeline-20260912-r1` |
| Policy SHA | `6a47be89b9b8693e0cf2104ea3db1ef30b5a5f2343e9ee857a0888fa9153de68` |
| Started SHA | `0e95d344dd7f1d546b2b32d80c41d162332bbef055fc765e4df49338f0be4b2b` |
| PID / creation time | `26652` / `1789235922.8164277` |
| Session | `24901` |
| Source store | `native-spine-source-completion-20260912-r1/complete-body-store` |
| Required source bodies | 31,166, all admitted with exact raw coverage |
| Maximum new exchange / parent jobs | 8,192 / 8,192; unchanged 128-job invocations |
| Wait bound / poll interval | 48 hours / 30 seconds |
| Evaluation allowance | 400 fresh answer streams, 200 logical judgments |
| Bound implementation files | 115 |

Configuration:
`.tmp/native-spine-full-corpus-pipeline-20260912-r1.json`. The policy contains the
resolved paths, source and attention-method hashes, six predecessor policy/start
bindings, execution bounds and the unchanged native evaluation policy.

The prerequisites are original ingestion PID 56400, source completion PID 54564,
expanded exchanges PID 49268, attention PID 42700, expanded parents PID 54716 and
R5 vectors PID 17192. Each identity includes its recorded creation time; PID
reuse alone cannot keep a completed predecessor alive. Current GPU caches are
reused only after their own complete result is available.

The stage order is:

1. Authenticate the completed JSON-aware store and prepare all exchange inputs.
2. Compile summary-only Qwen exchanges, reusing the completed R4 exchange cache.
3. Prepare attention inputs in a separate process, then compute the missing
   windows with the existing local Qwen method and cache.
4. Compile parent summaries, reusing the completed expanding parent cache.
5. Prepare full-corpus BGE vectors from the completed R5 cache, then encode new
   summaries using the existing FP32 CUDA batch-eight configuration.
6. Prepare the unchanged native full100 experiment, requiring all 100 separate
   histories to contain at least 1M eligible actual raw body tokens.
7. Run 400 fresh serial answer streams, then up to 200 logical Sol judgments.
8. Replay the joint report with no new provider calls and require identical SHA.

Full Qwen generation, prefix attention, BGE and evaluation run in successive
child processes. This releases each model before the next stage and keeps the
native Qwen runtime path out of the attention and evaluation processes. Each
child has a distinct execution reservation, request, actual PID/start record,
log, exit record and bound result. Windows children use hidden windows and a
structured argument list without shell interpolation.

The evaluator still checks that other Python jobs are absent before timed calls;
its ancestors are excluded by the existing check. The coordinator is then only
waiting for its child. Do not start concurrent GPU work or Python diagnostics in
this worktree once the `evaluation` stage has begun. Stage logs and receipts can
be inspected read-only through PowerShell.

The final handoff receipt copies the actual joint report's accuracy and gate
result. Completing all stages cannot turn a failed quality or latency result into
a pass. The benchmark must still achieve at least 95/100 and all eight matched
median/p95 TTFT/total-latency ratios at most 1.10 in the same fresh run.

## Verification and current limits

Fourteen new pipeline checks passed in 4.05 seconds (tool chunk `b5c998`). They
cover live and reused PIDs, strict terminal receipts, bounded resident merge
generation, partial/stalled compilation rejection, early incomplete-store
rejection, child execution without a shell, stop-on-failure ordering, refusal of
a second execution, and preservation of a failed joint gate. They use fixtures
and do not claim full-corpus model behavior or answer accuracy.

Actual preparation completed without releasing a model. Actual execution is
live and waiting for all six predecessors; `released.json` and the `stages/`
directory do not yet exist. All 115 pipeline-bound files and all 113 files bound
by the current full100 readiness result match their recorded hashes. Existing
source, repair, attention, hierarchy, vector and evaluation implementations were
not edited for this handoff.

The complete store, full-corpus model stages, real expanded namespace admission,
and fresh native full100 result remain pending. Inspect existing process handles
and stage journals on continuation; do not start another pipeline instance.
