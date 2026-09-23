# Matched engineering control gateway failure

**Status**: CURRENT — blocked before the control's first new engineering answer
**Date**: 2026-09-22
**Applies to**: `eval_results/paired-engineering-quality-20260922-r1`
**Depends on**: [Research Log 240](240%20-%202026-09-22%20-%20Real%20engineering%20session%20replay%20outcome.md), [Research Log 241](241%20-%202026-09-22%20-%20Engineering%20session%20token%20savings.md)

The user requested an engineering-quality comparison against direct full-context
use of the same model, assuming equal latency at the earlier seconds-level rate.
The completed memory continuation can be reused: all 73 responses used identical
actor instructions and the same tool policy. A fresh memory run was not launched.

The full-context control forks at the exact pre-edit checkpoint of that recorded
continuation: 311 chronological rows, step 3/action 28, and unchanged code at
`03105c423bf19b7a2b03f29a6b55e527f17ab2db`. The first three discussion prompts and
33 responses are shared history. Only the remaining continuation requires new
control generation. The fork's history hash and file hashes were verified.

The model alias, remaining original user prompts, 240-line reads, nine-module test
allowlist, eight-action batches and 80-response per-user limit match the memory
arm. System wording changes only to describe full chronological context in place
of retrieved memory. No original future solution or hidden evaluation is supplied
to the actor. Full context retains the conversation and tool results, omitting
duplicate internal memory activity receipts. It is never silently compacted.

## Result so far

| Measurement | Memory continuation | Full-context control |
| --- | --- | --- |
| New model calls for the memory arm | 0; existing result reused | Not applicable |
| Shared normalized behavioral checks | 7/8 passed | Not measured |
| Legacy `now=` argument compatibility | Failed separately | Not measured |
| New engineering answers in this comparison | Existing 73 reused | 0 |
| Latency | Assumed equal at the earlier seconds-level rate | Same assumption |

The normalized behavior suite replaces the original check's deprecated `now=`
argument with a clock patch through the current public interface. The other seven
checks are unchanged; API compatibility is scored separately. Both suites were
sealed before the control launched. The original memory evaluation's frozen 6/8
result remains unchanged. Its MCP current-energy display is the remaining failure
in the normalized behavior suite.

The first control request contains **346,026 locally estimated input tokens**,
282 messages and 1,309,900 UTF-8 bytes of message content. Three standard attempts
failed with `InternalServerError`. One separately recorded diagnostic using the
same full request returned HTTP 500 with this gateway reason:

```text
litellm.InternalServerError: InternalServerError: OpenAIException - Internal Server Error.
Received Model Group=codex_sdk/gpt-5.6-sol
Available Model Group Fallbacks=None
```

A tiny request to the same alias returned `OK`. The gateway lists the alias but
denies access to `/model/info`, so neither the supported input limit nor the
underlying server cause is established. This suggests a problem handling the
large request; it does not establish whether the cause is a context limit,
transport handling or another server fault.

**No engineering-quality comparison is available yet.** An unavailable control is
not a failed engineering solution and provides no quality advantage to memory.
The control executed no new file tools; its code still matches the common initial
checkout. Both workers have exited, and `full/STOP` is present.

## Evidence and next action

- [Sealed comparison plan](../../eval_results/paired-engineering-quality-20260922-r1/comparison-plan.json)
- [Current comparison status](../../eval_results/paired-engineering-quality-20260922-r1/comparison-status.json)
- [Shared memory behavior checks](../../eval_results/paired-engineering-quality-20260922-r1/quality-memory/pytest.log)
- [Exact full-context request](../../eval_results/paired-engineering-quality-20260922-r1/full/steps/03/actions/028/request.json)
- [Detailed server error](../../eval_results/paired-engineering-quality-20260922-r1/gateway-diagnostic/result.json)
- [Small-request probe](../../eval_results/paired-engineering-quality-20260922-r1/gateway-diagnostic/small-result.json)

Request SHA-256: `2d30e4f196f70efab4c730e15c459ca3e860308996deae061ef61db1e8aaa27b`.
The request file was written at **2026-09-23 01:09:20 UTC**. The user was asked for
the central-dev route's underlying log error or supported input limit. That
information is needed before deciding whether the exact control can resume.
Do not silently switch models, shorten its history, repeat the memory episode,
or treat this failed launch as a completed paired result.

## Verification

```powershell
$resultRoot = 'eval_results/paired-engineering-quality-20260922-r1'
Get-Content "$resultRoot/comparison-status.json" | ConvertFrom-Json |
    Select-Object status, new_memory_generation_calls, new_control_engineering_responses, workers_closed
Get-Content "$resultRoot/gateway-diagnostic/result.json" | ConvertFrom-Json |
    Select-Object status_code, error_type, body
```

Expected state: `blocked_full_context_gateway_error`, zero new memory calls,
zero control engineering answers, and all workers closed. Diagnose the gateway
before resuming; the engineering comparison remains unresolved.
