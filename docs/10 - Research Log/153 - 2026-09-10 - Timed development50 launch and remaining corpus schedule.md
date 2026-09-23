# Timed development50 launch and remaining corpus schedule

**Date:** 2026-09-10  
**Status:** timed comparison running; sixth-memory raw calls complete; remaining raw corpus scheduled afterward  
**Predecessor:** [152 - Five-memory reader and whole-summary comparison](152%20-%202026-09-10%20-%20Five%20memory%20reader%20and%20whole%20summary%20comparison.md)

## Timed comparison has started

The prepared three-arm, fifty-question comparison is now making its fresh
answer calls. Session **36114**, process **57888**, remains the authoritative
handle. Do not start another copy. The scheduled process transitioned from
waiting to execution after the raw-ingest dependency finished.

Both fixed readiness probes passed:

- Qwen: observed 12:46:24 UTC, 1.399 seconds.
- Terra: observed 12:46:27 UTC, 5.656 seconds.
- Readiness report:
  `75b320c05e584d90a3a86dc74454eb61bfbff33fa836c1422f06f538dbe741bc`.
- Timed release:
  `567de5aa07eca54e60f379cd0f025eccc6073cdd01c21c07fdce73663364524f`.

The root is
`eval_results/full1m-spine-combined-reader-development50-20260910-r1`.
The runner authenticated the completed bulk responses by zero-call replay
before releasing timed answers. The first API controls were observed complete.
No completed fifty-question score is claimed at this checkpoint. The most
recent complete scored comparison is still source diversity at **35/40**.

The complete frozen experiment remains **400 fresh Terra answers**, with at
most **150 logical Sol judgments**. Its implementation, evidence controls,
reader policies and success thresholds are unchanged from Log 152. A full100
accuracy and latency pass is still outstanding.

## Sixth-memory raw completion

Original raw-ingest session **65309** finished with **exit code 0**. All
**834 requests** completed, with 834 new calls and zero replay hits in the
original execution. The final raw-validation artifact is:

`eval_results/full100-spine-corpus-20260909-r1/offset-050/atoms-prefix-0834.json`

SHA-256:
`4fee9b92d633bb00e329633896af90c1ba2c8a90b2a024330cef824ebd5519e3`.

The old extractive validator reports 2,197 accepted atoms and 551 invalid
batches, giving `partial_or_invalid`. This does not mean that raw requests are
missing. Semantic source admission is a separate pending stage, as in the
previous complete memories. No claim is made yet that the sixth memory has
admitted atoms, attention leaves, or complete summary indexes.

After timed work finishes, use the full source-admission v3 audit over all 834
responses, then the existing bounded compaction/recovery and v8 verification
path as required. Preserve the original raw responses and reservations.

## Remaining four raw namespaces

All remaining raw prompts were reauthenticated with zero provider calls.
Preparation session **58675** completed with exit code 0. The new continuation
uses the existing executor and its original per-namespace preflights:

| Offset | Requests | Original execution preflight SHA-256 |
| --- | ---: | --- |
| 060 | 792 | `ab7175b56e2add9c25ef130bffa719b8672a66134d5e240fddd63513056e8194` |
| 070 | 838 | `8e596831452f1ee43f26c8ad34e5696029eed47782d23072628386986a459d5a` |
| 080 | 830 | `9bf47fb377dfbaeba5e106746e7c7c7f15f4a7d7500e34132ece7b975bdb6fc5` |
| 090 | 868 | `3351148ffe5140ea0840d15ef0764e1e12ff098da47c7c72aba94415764c98a8` |

Root: `eval_results/full100-spine-remaining-corpus-20260910-r1`.
The continuation preflight is
`ede7c1f12a5ba03d7924a38aaae7fd39da618a14d7d52f52779750638aa69edd`.
The fixed population totals **3,328 Terra-only raw-summary requests**. No raw
content is sent to Qwen. Namespace order is 60, 70, 80, 90, with four concurrent
requests within one namespace and zero automatic retries. A failed namespace
stops later namespaces and preserves original journals for diagnosis.

`start_after_timing.ps1` is running in session **84398**. This is a PowerShell
waiter, not another Python workload during the timed comparison. It watches
process **57888**, exact UTC creation ticks **639246399537879171**, and launches
no model work while that process is live. A completed campaign must also
authenticate the expected report before even the two synthetic readiness
probes can run. Failed or partial timed work cannot release bulk ingestion.

Starter script SHA-256:
`d5f1b128c60250de41a7885bf9e53e4ebf138ed927dfe0ed6e11331f9c27f751`.
Its live-process check was exercised against the actual dependency, a changed
creation identity, and an absent process. All three checks behaved correctly.
The script passed PowerShell parsing. The continuation runner separately
passed **6 scheduler tests in 1.90 seconds**:

```powershell
.\.pixi\envs\dev\python.exe -X utf8 -m pytest -q `
  eval_results/full100-spine-remaining-corpus-20260910-r1/test_runner.py `
  --basetemp eval_results/test-tmp-remaining-corpus-runner-r1
```

The waiter and local-gateway continuation were authorized and launched
successfully. Do not launch these namespaces independently while session 84398
is active. The readiness requests for this continuation are prepared but have
not run at this checkpoint. Even a completed raw continuation will not certify
source admission, hierarchy construction, or the full100 target.

## Analysis ready for the completed comparison

The campaign now contains `assess_results.py`. Run it only after its
`complete.json` exists and the timed process is terminal. It verifies all
400 saved answer responses and their 150 judgment bindings, reproduces the
reported accuracy and latency, and compares:

1. Current evidence with v2 versus the same evidence with v3.
2. v3 on current evidence versus v3 with whole-summary source coverage.
3. The complete candidate versus the current control.

It also reports prompt and answer token proxies, visible stream events, local
preparation time, API time and finish reasons. The repeated current control on
the first forty questions is compared with the previous 35/40 result to expose
fresh-answer variation on the unchanged prompts. Historical answers are used
only for post-score diagnosis, never as current predictions or route selectors.

This analysis script has passed syntax checking; it has not run on real results
because the complete comparison is still in progress. Do not infer a score or
latency improvement from its presence. Confirmation remains unopened by this
continuation, and the full joint target remains active.
