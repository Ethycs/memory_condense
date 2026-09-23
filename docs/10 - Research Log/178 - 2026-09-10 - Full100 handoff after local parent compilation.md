# Full100 handoff after local parent compilation

The full100 evaluation is now scheduled behind the live four-summary compiler.
It waits for that exact process to terminate, verifies all ten complete parent
hierarchies, waits for an idle workspace, prepares every comparison prompt,
checks reader/judge inference, and runs the existing fresh joint evaluator.
No answer calls have been sent while compilation remains active.

`tools/run_hierarchy_after_local_compilation.py` binds compiler PID 32784 and
creation time 1789098120.5012507, the measured batch result, imported summary
population, common parent method, source full100 preflight and implementation
hashes. The compiler is active in session 65191. The handoff is active in
session **61096**, last observed waiting on that process.

The scheduler requires exactly one complete, bound ten-memory population from
the current compiler. Every offset must use the expected output root and parent
method. The existing evaluator then checks each hierarchy against its original
million-token memory. A terminal process alone is insufficient: missing or
incomplete parents stop the handoff before model preparation or provider calls.

The subprocess sequence is:

1. Prepare all 100 questions with the original flat control and live Qwen
   hierarchy routing, preserving exact raw hydration and matched API prompts.
2. Make two bounded synthetic readiness calls, one to Terra and one to Sol.
   These contain no benchmark questions or raw corpus. The broken remote Qwen
   alias is irrelevant to this local-Qwen query path and is not a readiness gate.
3. Run the frozen evaluator's 400 fresh serial answer streams and subsequent
   200 logical judgments. Quality and latency use the same measured answers.
4. Replay the sealed judge outputs with zero new provider calls and bind the
   final evaluation report into the handoff completion receipt.

Each readiness call and the handoff itself has a single-use execution
reservation. Failures preserve existing requests and responses; no automatic
retry is performed. Waiting has a 72-hour operational deadline and ten-second
poll interval. Expiry does not terminate or restart a still-live compiler.
An identity mismatch or missing PID establishes that the originally bound
process is no longer active; access/inspection errors do not establish terminal
state. The idle check excludes the scheduler's own parent process when the
evaluation subprocess runs.

Nine tests pass in 1.92 seconds (chunk `f23592`). They cover exact process identity,
live wait expiry, observed terminal state, incomplete-population rejection,
bounded synthetic readiness and no resending, ordered preparation/answer/replay
handoff, dependency failure before calls and the execution flag. The tests use
fake subprocesses/providers; the existing evaluator's separate tests cover its
full 400-stream synthetic execution and quality/latency gates.

Preparation initially rejected a process identity because PowerShell rounded an
unquoted floating-point creation timestamp. Quoting the timestamp preserved
its exact value; the live-process check then passed. The strict comparison was
retained. No compiler process was restarted and no provider call was involved.

Handoff root: `eval_results/full1m-hierarchy-after-local-20260910-r1`.
Preflight SHA:
`ed42132fa8fd6a0c2ddee627b5136ab5000308d2c78d56dbbabf2a56c2ce7e92`.
Scheduled evaluation root:
`eval_results/full1m-hierarchical-spine-joint-full100-20260910-r2`.
Source control:
`eval_results/full1m-spine-relative-reservation-full100-20260910-r1`.

The local gateway execution was authorized and the scheduler was started with
network access for the eventual bounded evaluation. Its initial live output is
`waiting_for_compiler_pid: 32784` (tool chunk `6375a8`). Continue polling the
existing handles; do not start duplicate compilers or handoffs after a yielded
observation. Avoid other Python/model work during the eventual timed phase.

The target remains active and unmet. Neither scheduling the evaluation nor
successful parent-summary validation establishes 95% answer accuracy or
API-like query latency.
