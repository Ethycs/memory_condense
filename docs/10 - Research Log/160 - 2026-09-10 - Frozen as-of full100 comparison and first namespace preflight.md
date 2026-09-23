# Frozen as-of full100 comparison and first namespace preflight

**Date:** 2026-09-10  
**Status:** full100 protocol frozen; first 50 of 500 requests prepared; no answers or judges sent  
**Predecessor:** [159 - As-of summary routing and exact dated hydration](159%20-%202026-09-10%20-%20As-of%20summary%20routing%20and%20exact%20dated%20hydration.md)

The next intended timed comparison tests the date cutoff against semantic-seed
routing. Both methods use the same v2 reader, 3,072-token raw evidence budget,
and 256-token output cap. This is a separate full100 experiment, preserving the
unexecuted r3 source-diverse-versus-semantic-seed comparison and all its
prepared requests. No earlier prediction is reused, and no date-aware accuracy
gain has yet been measured.

The cutoff-only variant is selected before any new answers. The development50
packet audit showed that it already recovers the missing planting statement;
the extra relative-date hint changed only one packet and was unnecessary for
that recovery. This selection is an examined-development decision, not a
confirmation result. The full objective remains at least 95/100 with both API
latency baselines passing on the same fresh predictions.

## Frozen experiment

Root: `eval_results/full1m-spine-as-of-full100-20260910-r1`.

Protocol SHA:
`09e2b5c8c69a509bc23b2f62e24e0cb53507e30389127db1e6e7d67797ba29ad`.

The five requests for every question are a short direct API request, a live
semantic-seed answer, its identical-evidence API control, a live as-of answer,
and its identical-evidence API control. Memory/API pairs are adjacent, their
order alternates, and the three request groups rotate. All ten complete
approximately 1M-token memories and all 500 requests must be prepared before
release. Execution is serial with no automatic retries. At most 200 logical
Sol judgments follow the complete answer population.

The gate requires at least 95 correct answers on one complete method. Median
and p95 visible TTFT and total response latency must each stay within the
provisional 1.10 ratio against both that method's identical-evidence API
requests and the short API requests. The report cannot combine accuracy from
one method with latency from another, omit namespaces, or relax the allowance.
The streamed answers supplying latency are the answers subsequently judged.

The complete source admission remains common method v11:
`1b6ba1149c3962eb3a16e1fb55b6b2e16d8db811e274f9dc3ff21db3fe4d601a`.
Leaf and passage policies remain those already shared by the seven complete
memories. Raw recovery still accounts for the six original unknown attempts
and 403 preserved responses at offset 060.

## Implementation and input binding

- `tools/evaluate_spine_as_of.py` prepares and runs the two live routes with the
  same reader. It validates the exact original semantic-seed control prompts,
  question identities, complete ten-question population, counterbalanced call
  order, budgets, and unchanged memory inputs. Query embedding, routing, date
  projection, and hydration run inside the live request clock. Frozen prompts
  supply API controls; cached query vectors and answers do not supply live arms.
- `tools/report_joint_spine_as_of_full100.py` authenticates complete source
  admission, summary indexes, prediction/judgment bindings, and measurements
  across all ten namespaces. Judgments replay without new calls before the
  full100 gate is reported.
- `tools/run_spine_as_of_full100.py` requires complete bulk ingestion, fresh
  readiness, an idle worktree, and all 500 answers sealed before any judging.
  A failure preserves reservations and stops the sequence.
- `tools/prepare_spine_as_of_full100.py` prepares only prompts and bindings. It
  reads the existing r3 prepared namespace controls and leaves their files and
  compilation scheduler unchanged. For the first fifty questions, fresh as-of
  prompts must also reproduce the sealed diagnostic prompts from Log 159.

The source campaign protocol is still
`87f5d9ed7314a5341479f2ef2d8357ecb62ff95be6f2e7ecb7c28b7315bfefe3`.
The new protocol binds it and the development50 diagnostic
`abe09db4fa11b862b0d5347e99d49f34ea0dc67eae435be841b19982334da9e4`.
The old r3 campaign has not executed answers. Its prepared controls remain
useful inputs to this new experiment; it is not the next intended timed run.

## First complete-memory preparation

Offset 000 completed in session **67091**, exit 0. Its complete memory contains
1,041,276 token proxies. All ten semantic-seed control prompts reproduce, and
all ten as-of candidate prompts reproduce the earlier diagnostic under fresh
local query encoding. Fifty requests are prepared; none have been sent.

- Namespace preflight:
  `namespaces/offset-000/preflight.json`, SHA
  `9987cab305f6d77bc1b99e527eb97845e0842713778c945e513d9ed77759141b`.
- Campaign binding:
  `prepared/offset-000.json`, SHA
  `4c6ed5e0b678f8230d5c05e109330b7b61fda3ada1c14e4ede24e236fde73315`.

The actual full100 runner revalidated this preparation and stopped at the
missing `prepared/offset-010.json`. It created neither a runner plan nor an
execution reservation and sent no answers (exec `cb96ab`). Independent file
inspection found zero answer request journals, answers, or judge preflights
under the new root (exec `739c5a`). This is preparation progress, not a partial
full100 pass.

The evaluator, runner, bulk dependency, control-binding, and joint-gate suites
passed **31 tests in 73.75 s**, session **44045**, exit 0. The tests exercise
live routing before provider calls, changed-prompt rejection, identical reader
and API controls, 500-answer sealing, incomplete populations, preserved
failures, readiness and idle guards, complete raw recovery accounting,
unchanged historical control prompts, slow p95, and both latency baselines.
Log 159 separately records the 22 date-routing tests and real development50
packet audit.

## Live continuation

Raw session **42892** and compilation scheduler **12213** remain live. At
**16:31:41 UTC**, offset 070 had **753 response files / 756 request files** of
838. The same two handles were successfully polled; neither was restarted.
The scheduler still owns compilation of offsets 070/080/090 and prepares the
original r3 namespace controls after each complete memory. Its final step only
prepares the old runner; it does not send answers or judges.

Only the first new namespace was prepared in this turn so the local GPU is
available when offset 070 completes. After its compilation finishes, prepare
the remaining available date namespaces using the same frozen helper:

```powershell
.\.pixi\envs\dev\python.exe -X utf8 -u -m tools.prepare_spine_as_of_full100 namespace --output-root eval_results/full1m-spine-as-of-full100-20260910-r1 --offset 10
```

Each namespace must run in a separate process. Preserve an existing preflight;
the helper rejects overwriting it. Remaining offsets at this point are
010/020/030/040/050/060 plus 070/080/090 as their original controls finish.
Avoid competing local Python/GPU work when the original completion worker
starts; its capacity guard intentionally stops on conflicting work.

Once all ten new namespace bindings exist, use
`tools/run_spine_as_of_full100.py prepare` on the new root. Only after bulk and
compilation finish, fresh readiness passes, and the worktree is idle should
its `run` phase execute the 500 answers and subsequent judgments. Do not run
the old r3 answer campaign automatically. Confirmation remains unopened by this
continuation, with Log 102's historical exposure qualification unchanged.
