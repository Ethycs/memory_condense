# Engineering session token savings

**Status**: FROZEN
**Date**: 2026-09-22
**Applies to**: `eval_results/native-spine-engineering-live-session-20260922-r1`
**Depends on**: [Research Log 240](240%20-%202026-09-22%20-%20Real%20engineering%20session%20replay%20outcome.md)

The recorded engineering episode used **approximately 91% fewer generation input
tokens** than resending the same observed history at every model response, after
including recorded raw-summary and Qwen-summary inputs. This is an estimated
token comparison over one observed trajectory. No full-context control was run.

| Whole recorded episode, 106 actor responses | Input-token proxy |
| --- | ---: |
| Estimated repeated full history, excluding memory-only activity receipts | 49,821,450 |
| Actual coding-model requests | 3,337,259 |
| Recorded raw-summary requests | 1,000,838 |
| Recorded Qwen-summary requests | 278,230 |
| Total recorded generation inputs | 4,616,327 |
| Reduction against the history estimate above | 90.73% |

Counting internal activity receipts in the history estimate instead gives
52,185,080 tokens and a 91.15% reduction. The lower baseline above avoids charging
the hypothetical full-history system for those memory-only receipts.

For the **73 live continuation responses**, the coding model received a mean
**39,475 input tokens**, versus an estimated **610,440** history tokens per call
after excluding activity receipts: a **93.53% reduction in coding-model input**.
Summary overhead is reported for the whole recorded episode, not apportioned to
that live subset. The previous QA workload's roughly 1,537-token input average
does not describe this coding workload's bounded working conversation.

This answers the token-saving objective omitted from the initial outcome summary.
The other objectives remain distinct: original prompts were continued using memory
and bounded live work, but code test counts do not measure equivalent continuation
quality against full context. Retrieval averaged 2.46 seconds; ingestion took
176–554 seconds per successful live batch, and final persistence failed. Therefore
the token saving does not establish the 95%-accuracy/approximately-API-latency goal.

Counts use the local `cl100k_base` input proxy, not provider billing. Summary totals
include recorded retries and failed final persistence, but exclude prior reused
seed-summary compilation, generated output tokens, embeddings and attention
processing. The counterfactual retains repeated reads and tool wrappers from this
observed memory run; a separate full-context agent could take a different path.

The calculation tokenizes each journal row once and uses prefix sums at the
106 audited request boundaries. Its aggregate matches the sealed report's
full-history estimate exactly. No model calls, candidate changes or new tests
were required. Per-action counts and assumptions are retained in
[token-accounting.json](../../eval_results/native-spine-engineering-live-session-20260922-r1/token-accounting.json).

## Verification

From the worktree, inspect the calculation and its digest:

```powershell
$resultRoot = 'eval_results/native-spine-engineering-live-session-20260922-r1'
Get-Content "$resultRoot/token-accounting.json" | ConvertFrom-Json |
    Select-Object -ExpandProperty metrics
Get-FileHash -Algorithm SHA256 "$resultRoot/token-accounting.json"
Get-Content "$resultRoot/token-accounting.json.sha256"
```

Use the recorded input reduction to assess context-token savings; use the separate
engineering and persistence results to assess whether that saving preserves a
usable continuation. Do not substitute the code test pass count for memory accuracy.
