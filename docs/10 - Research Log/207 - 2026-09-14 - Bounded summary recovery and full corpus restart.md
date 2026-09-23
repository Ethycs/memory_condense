# Bounded summary recovery and full corpus restart

**Status:** The actual failed exchange label passes its original token limit.
The full-corpus restart described below was subsequently stopped when the user
deferred 100-history evaluation until the design is finalized. See
[Research Log 208](208%20-%202026-09-14%20-%20Defer%20full100%20until%20design%20is%20finalized.md).
No native full100 answer accuracy or serving-latency result exists.

## Fix and real reproduction

The ordinary Qwen refinements and the separately recorded single-row retry in
[Research Log 206](206%20-%202026-09-14%20-%20Full%20corpus%20ready%20and%20exchange%20length%20blocker.md)
all retained too much detail: the final label contained 129 tokens against a
128-token cap. Repeating that deterministic prompt did not resolve the problem.

`tools/native_spine_bounded_journal.py` adds two explicitly journaled abstraction
variants after ordinary refinements are exhausted. They request a routing label
of at most 16, then 8 ordinary words, replacing lists, identifiers and examples
with their general category. The model receives the same user spine and child
summaries. Original summaries and exact raw pointers remain stored separately.
The parser still enforces the original output cap, and an accepted response
must reach EOS. Generation is greedy, single-row and bounded to 128 new Qwen
tokens. No raw source text, occurrence-date metadata or remote call is added.

The actual failing merge key is
`b6a34579a998dce948393adb8f65f22eafc2346c665deb328259d0591526ac4c`.
Driver `.tmp/check_bounded_native_label_20260914_r1.py` tested that exact job:

- One local call using variant 3 succeeded, producing **22 tokens** under the
  unchanged 128-token validator. Generation took **5.0657 seconds** after load.
- Replay accepted the same response with no generation. Every original request
  and response remained byte-identical.
- The process exited zero. This is an offline summary repair measurement,
  not a query-latency or accuracy benchmark.

Artifacts are in `eval_results/native-spine-bounded-label-check-20260914-r1`.
Preflight SHA:
`3dfe708f608f2969f167d0234490846ca3a44b0a4056a0b1af58315b92f3e06d`.
Result SHA:
`1ca2c758775b0aa150a1ff52b641bbeecc2916150852c86d911b6e63a1706273`.

## Preservation and validation

The new exchange, attention-admission, parent and namespace adapters have
separate producer identities. Existing producer implementations remain intact;
completed source, exchange, attention and parent artifacts retain their original
fingerprints. The evaluator and pipeline orchestrator select the new producer.
The evaluator's accuracy, timing, evidence and gold-access policies are unchanged.

Before modifying those two orchestration files, all 115 original implementation
files were verified and archived at
`.tmp/native-runtime-before-bounded-recovery-20260914-r1.zip`, SHA
`7727270cab2c7cc92c044f47ab479b6eb5841ece16a664d68a018f2811d504cd`.
The old pipeline policy is historical and is not rewritten to match new code.

The successor authenticates accepted outputs and completed attempt records from
the retired exchange journal, including the failed explicit retry. This prevents
regenerating accepted work or repeating exhausted ordinary refinements. A
reserved call without a response still stops execution. Prepared summary inputs
are copied and hash-checked; raw source ingestion is not repeated.

**74 focused tests pass**, covering bounded exchange and parent recovery,
interrupted-call handling, EOS and length rejection, source-journal preservation,
exact raw-span hydration, legacy parent-cache ancestry, attention contracts,
complete-population admission and fresh full100 evaluation controls. Command:

```powershell
.\.pixi\envs\dev\python.exe -X utf8 -m pytest tests/test_native_spine_bounded_recovery.py tests/test_native_spine_full100.py tests/test_native_spine_full_corpus_pipeline.py tests/test_expanding_native_spine_exchanges.py tests/test_expanding_native_parent_reuse.py tests/test_native_spine_attention.py tests/test_parent_budget_native_namespace.py -q --tb=short --basetemp=.tmp/pytest-bounded-r3
```

The first test launch hit Windows permissions on the shared system temporary
directory before setup. The worktree-local reruns passed. `git diff --check`
also passed.

## Full-corpus restart

New root: `eval_results/native-spine-full-corpus-pipeline-20260914-r1`.
Configuration: `.tmp/native-spine-full-corpus-pipeline-20260914-r1.json`.
Policy SHA:
`ab5ed34c2f36915caef5bae2b698224ecfe98f40bbe95193bb92408f5bf7f62f`.
Controller PID at launch: `61124`; use its sealed start record and creation time
to check liveness rather than trusting the PID alone.

The run uses the completed 31,166-body source store with SHA
`721ef21c4e1d439cb30346c1abf0a6c25f1b19cd61ff52f8ce51f2969bd7e267`.
It imports the retired September 13 exchange journal and the complete exchange
R4, attention-cache R1, parent R1 and vector R5 caches recorded in prior logs.
All previous owners, including the real-label check, had exited before launch.

The original 8,192-job exchange allowance has **7,049 jobs remaining**, after
deducting 1,142 response-backed jobs in 290 prior batches and the one new label
check. There are no uncertain prior calls. The full compiler will independently
regenerate the checked label once under its own producer preflight; that call
counts against the remaining allowance. The parent allowance stays 8,192.
Budget receipt `.tmp/native-spine-bounded-restart-budget-20260914-r1.json` has SHA
`0ae82d9d7af816c374aff5cc8f2339dc40b2e62a78b14f9b45063e571be5b783`.

The controller runs exchange compilation, attention, parent compilation and
vectors in separate sequential processes, followed by one fresh full100 and
zero-call report replay. No additional GPU job should run alongside it.

The target still requires at least 95% accuracy on 100 separate histories,
each with at least 1M actual body tokens, and all eight median/p95 TTFT/total
latency ratios at most 1.10 against the matched API controls in the same run.
The successful label repair does not establish either benchmark gate.
