# Five-memory reader and whole-summary comparison

**Date:** 2026-09-10  
**Status:** 400 answer requests prepared; scheduled after live sixth-memory ingest; no successor answers yet  
**Predecessor:** [151 - Source coverage development40 result and fifth memory recovery](151%20-%202026-09-10%20-%20Source%20coverage%20development40%20result%20and%20fifth%20memory%20recovery.md)

## Purpose and comparison

The latest scored successor remains source diversity at **35/40**, against
34/40 for passage routing and 33/40 for scoped terms. The whole-summary
supplement has recovered omitted evidence in the previous provider-free audit,
but its answer accuracy is still unmeasured. The existing v3 reader previously
helped some qualification and recommendation questions on different evidence.
This experiment separates those changes without introducing another prompt.

All fifty questions in the five complete memories are prepared before any
answer calls. Each memory contains approximately 1.04M token proxies. The
three memory arms are:

| Arm | Routing | Reader | Short API control |
| --- | --- | --- | --- |
| `base` | Current source diversity plus scoped terms | v2 | `base_short_api` |
| `reader` | Identical evidence to `base` | Existing v3 | `reader_short_api` |
| `combined_reader` | The same route plus whole-summary source coverage | Existing v3 | `reader_short_api` |

Each memory arm also has its own byte-identical evidence API control. This
produces eight answer calls per question: three live memory calls, three
matching evidence calls, and two short API calls. The cap is **400 fresh Terra
answers and 150 logical Sol judgments**. Identical judge requests may reuse
authenticated responses; answer predictions are never reused.

The reader comparison holds evidence fixed. The routing comparison holds the
v3 reader fixed. The shared short API control has the exact v3 policy used by
both v3 memory arms. Matched pairs remain adjacent, with memory first on five
questions and API first on five in every namespace. Group order rotates.

## Routing and measurement

`tools/spine_combined_memory.py` preserves the previous user evidence, then
adds user-containing sections from distinct sources in the top 32 whole-summary
matches, up to eight additional sources. It uses the already computed query
vector. Raw reads occur only during exact hydration, after summary selection.
The evidence limit remains 3,072 token proxies and 128 spans.

Qwen receives summaries only and contributes attention partitioning at ingest.
Live query routing uses BGE summary vectors; this experiment adds no query-time
Qwen call or embedding call. The comparison does not claim that query-time
Qwen attention has become the production router.

The new evaluator is `tools/evaluate_spine_combined_reader.py`. Every measured
memory call recomputes routing and hydration inside its clock and must reproduce
the prepared messages exactly before invoking Terra. Cold resident setup is
reported separately; it is excluded from warm query latency. The judged
predictions are the same streamed responses whose latency was measured.

`tools/report_joint_spine_combined_reader_full100.py` retains the complete
100-question gate, including common compilation/admission methods and both
short and identical-evidence API baselines. Accuracy must reach 95/100, and
median and p95 visible TTFT and total latency must stay within the provisional
10% allowance. A development50 report cannot pass that gate.

## Prepared artifacts and replay

Campaign root:
`eval_results/full1m-spine-combined-reader-development50-20260910-r1`.

The complete preparation is SHA-256
`1db86b310ebbab4c2e90cdd6c8bf77aec11cab491362d699993979fad24cfeb5`.
Preparation session `11956` completed with exit code 0. No provider calls were
made during preparation or admission replay.

Namespace roots follow
`eval_results/full1m-spine-combined-reader-joint-offsetNNN-20260910-r1`:

| Offset | Questions | Answer calls | Preflight SHA-256 |
| --- | ---: | ---: | --- |
| 000 | 10 | 80 | `fea98a0595def9c51480cb2aabc7028ade3dc1f463de261acfafecd376fff025` |
| 010 | 10 | 80 | `99fda39b119136669fc557bca191a2faf9c96140f6f058358690b9dc2d142ad8` |
| 020 | 10 | 80 | `a49bdb141f24abbbd6fb5fe5c0df9eeb907b2715c10a56a4c87e1237981264ad` |
| 030 | 10 | 80 | `db6aca876d53126e68e4d9de73fa8f545386c95746926edce293f0bab50143bc` |
| 040 | 10 | 80 | `1598da61ccc156d876435fae6fe22592365ea00747164ebf7c7cc41e0233bf84` |

All forty previous source-diverse control messages and short controls reproduce
exactly. All forty combined candidate evidence messages match the earlier
whole-summary audit; only the intentionally changed reader system message
differs. The fifth memory contributes ten additional development questions.
Preparation reads no predictions or gold answers.

All five complete memories replay under these common policies:

- Attention leaf compilation:
  `06c51b3892c2ecebc3155003b3ec6906d62b12eb6f55b7095eb571787f94e878`.
- Complete user-summary passage addresses:
  `bf856860fc173039fe26ef53dcf88090495fa20bfdb271ca19d5739ed3f48816`.
- Conditional source admission v8:
  `92187190058445ed4fdbd8375fa870162117fa1b1b913f2aeb2b7c9bc2bcabcf`.

The previous atoms and saved responses remain unchanged. The new gate verifies
the same bounded recovery method across the older and fifth-memory artifacts.

## Execution dependency and current state

The serial runner plan is SHA-256
`f02906d6a17391876277e6ffe5720be1e98af0dee32e7c63bec20e980be6b28d`.
The live-process wait plan is SHA-256
`7480aa2f5046747166719a9701baa8ddb605ef2f1312b91c127bfbd769853f52`.

`wait_and_run.py run --enable-provider` is running in session **36114**. It
confirmed the identified ingest process live at 12:25:54 UTC. That process is
PID **54804**, creation time **1789039783.7666545**, tool session **65309**,
running offset 50 of the original corpus with 834 prepared requests. Its
execution preflight is
`920deec1229eaaea4c570f831d7132e524953550615ca897a8a724422664653c`.
The latest ingest poll in this continuation reached batch 622 and returned the
same live session. This is progress within the namespace, not a completion claim.

The waiter polls the exact process identity every thirty seconds. Process
absence alone cannot release answers: all 834 raw responses must have a final
artifact, and the serial runner authenticates them by zero-call replay. No
other local Python work may be active in this worktree when timing starts.
Two fixed synthetic gateway probes then provide the freshness requirement.
The readiness root is the campaign's `gateway-readiness` subdirectory.

The local-gateway execution was authorized and started successfully. It uses
zero automatic retries and preserves reservations after failure. A timeout or
failed dependency stops the campaign for diagnosis. Do not start a second
waiter, replay readiness requests, or restart the original ingest merely
because a tool observation times out.

The scheduled campaign must finish before further bulk ingestion or model
compilation is started in this worktree. After the timed run, continue admitting
offset 50 and constructing offsets 60–90 for full100. At this checkpoint,
there is no new answer score, no full100 target pass, and no confirmation run.

## Verification

The evaluator, live retrieval, exact API controls, full100 gates, passage
population verification and source-coverage checks passed **176 tests in
8.58 seconds**:

```powershell
.\.pixi\envs\dev\python.exe -X utf8 -m pytest -q `
  tests/test_joint_spine_reader_eval.py `
  tests/test_joint_spine_combined_reader_eval.py `
  tests/test_joint_source_spine_full100_gate.py `
  tests/test_spine_facet_full100_verification.py `
  tests/test_spine_source_coverage.py `
  --basetemp eval_results/test-tmp-combined-reader-r1
```

The new campaign scheduler separately passed **6 tests in 1.57 seconds**:

```powershell
.\.pixi\envs\dev\python.exe -X utf8 -m pytest -q `
  eval_results/full1m-spine-combined-reader-development50-20260910-r1/test_runner.py `
  --basetemp eval_results/test-tmp-combined-reader-runner-r1
```

These checks reject a changed live candidate before its provider call, reject
a mismatched reader even when its evidence API is changed with it, preserve
unacknowledged reservations, reject partial full100 populations, and prevent
unfinished or concurrent bulk work from releasing timed requests. They do not
substitute for measured answer accuracy or latency.
