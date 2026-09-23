# Real engineering session replay outcome

**Status**: FROZEN — coding replay finished; final memory persistence failed
**Date**: 2026-09-22
**Applies to**: `eval_results/native-spine-engineering-live-session-20260922-r1`
**Depends on**: [Research Log 239](239%20-%202026-09-17%20-%20Bounded%20engineering%20session%20memory%20evaluation.md), [documentation style guide](../../Agentic%20Technique%20Master.md)

The model completed all eight original prompts from a real engineering episode,
changed 25 files, and passed 237 regression checks plus 73 additional checks.
Six of eight frozen independent checks passed. It implemented turn-based decay,
retained the user's correction through later work, reconciled documentation, and
added benchmark instrumentation. These are actual generated edits and executed
tests in an isolated historical checkout.

The end-to-end lifecycle did **not** pass. Qwen repeatedly exceeded the final
summary budget, preventing the final memory installation. A separate process
reopened 481 stored turns matching the journal exactly; 43 final events remained
only in the durable journal. Successful ingestion batches also took minutes.
The application demonstrated useful coding continuity, but is not yet validated
as a reliable, interactive replacement for accumulated engineering context.

## Scope and context lifecycle

The checkout began at `03105c423bf19b7a2b03f29a6b55e527f17ab2db`, using the same
214-message historical seed as the earlier replay. Original pre-episode tool dumps
were absent. Original future assistant solutions and hidden acceptance checks
were withheld from the actor. This run continued the same episode, retaining the
33 recorded responses from the earlier attempt without replaying completed calls.
It added 73 responses, for 106 total.

The current user prompt and complete recent action/tool pairs stayed in a bounded
49,152-token working conversation, within a 65,536-token total prompt cap. Older
work entered normal `MemoryCondenser.ingest_many` before a new user prompt or
before un-ingested working context was evicted. The application persisted and
reopened its native memory before retrieval. Retrieval was cached between those
boundaries; this was not an ingestion-after-every-tool experiment.

Sol summarized raw observations and generated code. Qwen received typed summaries
for merging and user-spine attention; exact raw sections were hydrated after
summary routing. Reserved recent user instructions, the last completed reply,
and actual work receipts were also recovered through memory. Persistent working
files remained available for ordinary reads and edits.

The evidence audit passed for **424 packets and 3,040 exact spans**. This proves
source reconstruction, not semantic completeness of every summary. All 25 final
changed files match recorded successful model writes. **UNCOMMITTED:** the
candidate changes remain isolated; the operator did not repair the generated code.

## What the actor built

| Original prompt | Observed outcome |
| --- | --- |
| “What is day 14” | Explained the existing wall-clock decay evaluation. |
| Subsequent turns should assign decay | Identified the mismatch with the intended conversational clock. |
| “goldilocks zone” | Restated selective reinforcement and forgetting requirements. |
| “Real time shouldn't be the decay but rather per turn” | Implemented turn-based decay, a 32-turn half-life, schema migration, and once-per-turn reheating; ran tests. |
| “look at the comit history, did we drop anything?” | Reviewed history and found stale documentation plus evaluation-side reheating. |
| “Ok so how do we deliver the system that was specified” | Fixed evaluation-side reheating, reconciled documentation, and passed 237 relevant tests. |
| “Put it in docs” | Wrote the staged completion plan. |
| “Go” | Implemented benchmark token, usage, retrieval, generation and wall-time instrumentation with aggregates and CLI output. |

The 25 changed files comprise ten source files, five test files and ten documents.
The actor did not execute a benchmark dataset experiment. Its test tool exposed
nine offline modules and rejected its attempts to run `test_benchmark.py`.
Benchmark and related checks were therefore run independently after coding ended.
The complete patch and verbatim final replies are retained in the report artifacts.

## Validation

| Check | Result | Interpretation |
| --- | --- | --- |
| Candidate regression suite | 237 passed | Relevant existing and actor-modified tests pass. |
| Frozen independent acceptance | 6 passed, 2 failed | An API incompatibility and an actual energy-display defect remain. |
| Additional benchmark/related tests and clock diagnostic | 73 passed, 6 slow tests deselected | Includes one separate current-API wall-clock diagnostic; does not change the frozen acceptance score. |
| Historical implementation API compatibility | 84 passed, 34 failed | Diagnostic expectations include different parameter names and schema fields; these are not 34 independent functional failures. |
| Independent checkpoint before later prompts | 6 passed, 2 failed | Same failures as final acceptance; results were not supplied to the actor. |

The frozen wall-clock check failed before reaching its assertion because the
candidate removed `recall_memories(now=...)`. A separate diagnostic advanced the
clock by 1,000 days while calling the current interface and passed. This supports
the intended wall-clock independence without hiding the API compatibility failure.

**BUG:** MCP statistics showed `e=0.80` when current decayed energy was `e=0.40`.
The formatter called `item_energy(item)` without the current turn, even though its
heat-tier calculation used the current turn. The other independent checks covered
turn decay and reopen, pins, once-per-turn reheating, duplicate reinforcement,
untouched unrelated memories, and retention under frequent use.

## Timing and context size

Live-continuation measurements exclude the 33 inherited responses unless stated.
Token counts are local estimates; provider usage counters reported zero.

| Measurement | Observed value |
| --- | --- |
| Live actor responses | 73 |
| Mean / maximum actor input | 39,475 / 54,695 tokens |
| Input at the four subsequent user-prompt boundaries | 5,517; 5,515; 4,460; 4,251 tokens |
| Fresh live retrieval | 9 calls; mean 2.460 s; maximum 3.393 s |
| Successful live ingestion | 9 batches; mean 337.797 s; range 176.094–553.760 s |
| Failed ingestion and recovery elapsed estimates | 1,285.505 s, additional to successful ingestion |
| Live wall time from preparation through final stop | 7,527.185 s, approximately 125.5 minutes; includes repairs |

The report's all-action ingestion total, including inherited actions and failed
attempts, is 7,410.512 seconds. It has a different scope from the live-only wall
measurement above. Cached actions have zero recorded ingestion/retrieval time;
averaging those zeros into latency would conceal the boundary stalls.

The journal contains an estimated 820,786 raw tokens, including repeated file
reads and tool wrappers. That is neither a natural 1M-token engineering history
nor proof of 95% engineering accuracy. There was no matched full-context control.
The earlier question-answer scores describe a different workload.

## Runtime failures and final persistence

Two ingestion problems were repaired without changing candidate code or providing
human-authored solution hints. Oversized model-selected support quotes were
shortened to source-exact prefixes within the existing limit. Repeated pending
raw fragments were compiled once per cache key while preserving every original
source occurrence and address. Their adapters and recovery records are retained;
failed elapsed time is included in the reported totals.

After all eight coding prompts finished, the final user-spine summary merge
returned 135, 131 and 146 tokens against a 128-token limit. Two bounded recovery
attempts requested 96 and 64 tokens and returned 135 and 137 respectively. The
budget was not relaxed and the summaries were not manually truncated. Recovery
was exhausted and the final installation failed.

The last good native snapshot contains **481 of 524 journaled turns**. A separate
process verified its exact chronological prefix. The **43 uninstalled events**,
generated files and validation results remain available on disk. There is no
`complete.json`; `STOP` is present and all coding/gateway workers are closed.
The report status is `actor_complete_memory_save_failed`.

## Artifacts and next work

- [Recorded report and original-prompt replies](../../eval_results/native-spine-engineering-live-session-20260922-r1/report.md)
- [Machine-readable audited result](../../eval_results/native-spine-engineering-live-session-20260922-r1/report.json)
- [Generated candidate patch](../../eval_results/native-spine-engineering-live-session-20260922-r1/report.patch)
- [Final memory failure and separate-process reopen](../../eval_results/native-spine-engineering-live-session-20260922-r1/final-memory-failure.json)
- [Candidate write provenance](../../eval_results/native-spine-engineering-live-session-20260922-r1/candidate-write-audit.json)
- [Independent acceptance failures](../../eval_results/native-spine-engineering-live-session-20260922-r1/independent-behavior.log)
- [Additional validation](../../eval_results/native-spine-engineering-live-session-20260922-r1/additional-validation/pytest.log)

Priority order for subsequent development:

1. Make bounded summary compression and final persistence reliable; the recorded
   failed inputs provide a targeted regression case without repeating the episode.
2. Reduce or move ingestion work off the interactive path while retaining an
   explicit durability boundary; minute-scale stalls currently block the latency goal.
3. Correct MCP energy formatting and decide the compatibility policy for removed
   public arguments, then rerun the affected checks against a separately labeled fix.

No further generation, QA campaigns, candidate repairs or retries are part of this
completed evaluation. Review the failure before treating this run as deployment
evidence.

## Verification

From the worktree, inspect the result and verify its sealed digest without model
calls or replaying the session:

```powershell
$resultRoot = 'eval_results/native-spine-engineering-live-session-20260922-r1'
Get-FileHash -Algorithm SHA256 "$resultRoot/report.json"
Get-Content "$resultRoot/report.json.sha256"
Get-Content "$resultRoot/report.json" | ConvertFrom-Json |
    Select-Object status, complete_prompts, last_answer_ingested_turn_count, events_after_last_answer_ingestion
```

Expected digest: `c3e32b1388b51c01a10f15021bfc6835d5c9fdeb86d6882e4c391e062d74fb96`.
Expected state: `actor_complete_memory_save_failed`, eight completed prompts,
481 installed turns and 43 remaining events. Use that state to distinguish useful
coding progress from successful end-to-end memory persistence.
