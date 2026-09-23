# Automatic full100 handoff after complete ingestion

**Date:** 2026-09-10  
**Status:** handoff waiting on live dependencies; 400/500 requests prepared; no new answer result  
**Predecessor:** [162 - Exact token accounting and eight complete memories](162%20-%202026-09-10%20-%20Exact%20token%20accounting%20and%20eight%20complete%20memories.md)

A bounded handoff now owns the final preparation and release of the frozen
semantic-seed versus as-of full100 experiment. It waits for the existing raw
and compilation jobs to finish successfully. It then prepares offsets 080 and
090, validates all 500 requests, runs two fresh synthetic gateway probes, and
invokes the existing full100 evaluator. All answers must be sealed before
judging. Neither routing, reader, budgets, nor joint gates changed.

## Implementation and release

`tools/run_spine_as_of_after_compilation.ps1` waits without creating another
Python job in the worktree, preserving the compilation scheduler's local
capacity rule. It authenticates its sealed plan and its own source hash,
checks PID plus creation time, watches dependency failures, and requires both
complete payloads and sidecars after the processes end. The wait is bounded
at twelve hours with ten-second polls. Its exclusive reservation prevents a
second waiting process from using the same release.

`tools/run_spine_as_of_after_compilation.py` independently rechecks the pinned
raw and compilation plans/releases, original first80 preparation, frozen
runtime hashes, terminal process identities, both complete memory workers,
and their whole raw inputs. It also requires an idle worktree. It rejects
existing or partial preparation of either remaining namespace. Separate
preparation subprocesses release the query encoder before timed evaluation.

Readiness uses the existing two-call synthetic Qwen/Terra protocol. Its
responses must authenticate and remain fresh when the unchanged full100 runner
starts. The handoff itself makes no new raw-ingestion or summary-compilation
calls. The fixed limits are two readiness calls, 500 answer calls, and at most
200 logical judgments, with zero automatic retries. Any failure retains
earlier outputs and reservations and stops the sequence. The older r3 answer
campaign remains unexecuted.

Root: `eval_results/full1m-spine-as-of-after-compilation-20260910-r1`.

- Handoff preflight SHA:
  `1cff04aaf9be87e538bd5b9b088c97dc44ede18c968a894c4068bf34f6fae5ef`.
- Synthetic readiness preflight SHA:
  `0bfaad0e01542cb37ce9b176981cce34271f680b6e55b00d4fa9ef960791ff25`.
- Unchanged full100 protocol SHA:
  `09e2b5c8c69a509bc23b2f62e24e0cb53507e30389127db1e6e7d67797ba29ad`.
- Unchanged first80 preparation SHA:
  `43869f83742364a4d108faeaf2a297d6fd47c2aa699f2a4d3ed7ce12a76455cc`.

Preparation completed without calls in exec `7045ee`. The authorized escalated
launch succeeded in exec `17b196` and returned live session **13811**. Its
PowerShell PID is **61480**, created **2026-09-10T17:16:34.8494369Z**.
`wait-started.json` is an informational process observation, not a sealed
completion or evidence of any benchmark execution. The live session was
subsequently polled successfully in exec `70f41f`.

## Prepared prompt volume, before serving

A separate read-only audit authenticated the first80 preparation and counted
all 400 rendered prompts without loading an encoder, references, or answers.
Every memory prompt matches its identical-evidence API control exactly. Whole
chat token proxies include the reader policy and dated question:

| Arm | Median tokens | p95 tokens | Minimum–maximum |
| --- | ---: | ---: | ---: |
| Short API | 436 | 461.15 | 426–468 |
| Semantic seeds | 3,396 | 3,495.20 | 3,170–3,530 |
| As-of cutoff | 3,402 | 3,494.15 | 2,980–3,530 |

As-of prompts are smaller for 24 questions, equal in token count for 29, and
larger for 27. The mean paired reduction is only 8.0375 tokens. The cutoff
therefore changes evidence selection with little overall change in prompt
volume. This does not measure or predict a latency ratio; any speed or accuracy
gain still requires the frozen joint run. Keep that experiment unchanged.

Audit root:
`eval_results/full1m-spine-as-of-prompt-volume80-20260910-r1`.
Its `run.py` records its own source hash, tokenizer hash, original protocol and
preparation hashes, and every counted prompt identity. `report.json` SHA is
`d35eed0b86d805f2b1ceb68ffa8acd5efa8c345b66cb52fb1b2d26fcc0c79309`.
Execution `f67367` completed in 3.39 s with zero provider calls. The audit
asserts all eighty five-arm groups, exact API-pair identities, and the unchanged
5,500-token prompt ceiling. Equal token count does not imply equal evidence.

## Verification and current ownership

Twenty focused tests passed in 2.20 seconds (exec `a3891a`). They cover live,
missing and reused PIDs; inaccessible process state; changed raw/completion
bindings; missing memory completion; failed dependencies; partial preparation;
readiness failure; provider authorization; and preserved reservations after
failure. A separate actual PowerShell parse and live/reused/missing process
check passed (exec `ea3a9e`). The launched script also passed its actual
preflight-sidecar and source-hash checks before entering the wait.

Raw session **42892** and compilation session **48873** remain live under
their releases recorded in Log 162. Raw work is progressing in the ninth
namespace; the tenth is still outstanding. The handoff session **13811** owns
the later as-of preparation and answer/judge release. Do not independently
prepare offsets 080/090 or launch another answer runner while it is active.

Observe these existing handles and their output artifacts. If a dependency
fails, diagnose the preserved failure before constructing any successor;
an observation timeout alone is not terminal. After evaluation completes,
inspect `full1m-spine-as-of-full100-20260910-r1/joint-full100.json` and the
authenticated predictions/timings. Completion of the handoff does not imply
that the accuracy and latency target passed.
