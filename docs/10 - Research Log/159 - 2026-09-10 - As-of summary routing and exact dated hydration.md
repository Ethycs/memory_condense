# As-of summary routing and exact dated hydration

**Date:** 2026-09-10  
**Status:** implemented and audited on development50; answer accuracy and joint latency unmeasured  
**Predecessor:** [158 - Seventh memory admission and future evidence audit](158%20-%202026-09-10%20-%20Seventh%20memory%20admission%20and%20future%20evidence%20audit.md)

The date-aware candidate removes all 240 future-dated raw spans from the fifty
semantic-seed development packets and replaces them with eligible evidence.
All fifty frozen semantic-seed control prompts reproduce under fresh local
query encoding. The previously missing April 21 tomato-planting statement now
reaches the gardening packet. These are evidence-routing results, not judged
answers. The latest scored reader remains 41/50; no 95% or latency pass follows
from this diagnostic.

## Implementation and date boundary

`search/as_of_spine_routing.py` and `tools/spine_as_of_memory.py` add two separate
experimental routes. Both retain semantic-only initial seeds, the existing
reader, the complete admitted memory, and the 3,072-token / 128-span raw budget.
The original semantic-seed and source-diverse routes are unchanged.

The as-of route uses the bound question day and stored transcript timestamps.
Wholly future leaves are excluded before taking each channel's top candidates,
allowing older leaves to refill semantic, user, and passage routes. User source
backfill excludes later turns before spending its 2,048-token metadata
reservation. Source diversity and scoped term coverage still precede one final
raw hydration.

Mixed-date sources are not dropped. A mixed leaf remains addressable when it
contains any eligible span. Final projection retains only original raw-span
descriptors dated on or before the question day; a partial attachment gets a
new descriptor and receipt bound to its original section and retained spans.
The projected index describes the selected evidence, not the complete corpus.
Foreign or changed source descriptors cannot enter this projection. The
existing hydrator then verifies exact source, role, timestamp, coordinates,
text hashes, and final budgets.

Same-day statements remain eligible because the question timezone is not
specified. Event dates inside statements do not control eligibility: an older
statement about a future plan remains usable. Mixed-leaf summaries can still
describe later mentions and influence routing; the restriction here governs
the raw evidence delivered to the reader, not a reconstructed historical
summary index. No raw text is embedded or sent to Qwen by this change. Qwen
continues to partition summaries at ingest; this candidate makes no query-time
Qwen calls and uses the existing live query embedding inputs.

The optional second route adds `summary_time_prior_v2.py`. It recognizes one
unambiguous numeric or written relative expression, such as “two weeks ago,”
and uses a soft mention-date preference while retaining global fallback. Month
and year subtraction clamps to valid calendar days; ambiguous expressions
yield no preference. This is not a certified event-date filter.

## Real-data diagnostic

`tools/audit_spine_as_of.py` runs each complete namespace in a separate process.
It reproduces the frozen semantic-seed prompt, computes fresh vectors for both
candidates, authenticates hydration, and seals candidate prompts and route
receipts. It loads neither references nor predictions and makes no answer or
judge calls. All five memories still contain approximately 1.04M token proxies;
the date cutoff changes query eligibility within those complete memories.

Root: `eval_results/full1m-spine-as-of-development50-20260910-r1`.

Aggregate `audit.json` SHA:
`abe09db4fa11b862b0d5347e99d49f34ea0dc67eae435be841b19982334da9e4`.

| Measurement over all 50 questions | As-of cutoff | Cutoff plus relative hint |
| --- | ---: | ---: |
| Changed packets | 36 | 36 |
| Future raw spans before | 240 | 240 |
| Future raw spans after | 0 | 0 |
| Added eligible raw spans | 273 | 274 |
| Removed previously included eligible spans | 19 | 19 |
| Removed previously included eligible user spans | 11 | 11 |

All fourteen packets with no future spans remain byte-identical. Changes to the
36 affected packets can still displace useful earlier material under the fixed
budget. The eleven removed user spans occur in nine questions. Their loss is
recorded explicitly; the candidate does not claim preservation of all prior
user evidence or improved answer accuracy.

Local diagnostic routing/embedding/hydration medians were 0.177 s and 0.174 s.
Raw ingestion ran concurrently, no answer API was measured, and these timings
are not an idle-serving or API-relative latency result. Timing must be measured
again with fresh answers under the joint evaluation protocol.

| Namespace offset | Audit SHA-256 |
| --- | --- |
| 000 | `80dc1ff89124782cf7c63919eea22b2d8c2451def804929cc388a5c638acd7f1` |
| 010 | `86e4745bc063b64889db041843a7ec3c02c11e4d6661392e9cd9e9a377bd9083` |
| 020 | `02ec3465cd5a359a0e17cf2679d226c80ed5b2106d26c7a5df703fcccbb8ef29` |
| 030 | `0517b3c302735f86c69d7b8ee1fc5f432e343a6549217c0c5d9c9634c85292a0` |
| 040 | `591613d42488bfec0047cebac1eca463ac5cef6467f08daf2fe555ec8e271dea` |

### Gardening witness and relative-prior comparison

Question 43 asks, on May 5, what gardening activity happened two weeks earlier.
Both new packets hydrate the April 21 user statement about planting twelve new
tomato saplings. The cutoff alone recovers it; the relative hint is not needed
for that evidence recovery. The hint changes only this question's final packet
across all fifty comparisons. In the hint variant, the gardening-app statement
precedes the planting statement, so priority is not automatically better.

- Cutoff-only question-43 prompt: `5bf9b738671b8ef90a6b966e26939bbe2b84af8323c664d6b2b771b839157041`.
- Relative-hint question-43 prompt: `7229acd87b724761e44a0b5b9810ba3504dcac1bd451f6ade6262620fa7e48cc`.

These question ordinals and witness terms were inspected after all candidate
packets were sealed. They do not select production routes or supply answers.
The evidence supports testing the simpler cutoff candidate first; the separate
relative variant remains available for a later matched comparison.

## Checks, reproduction, and live work

`tests/test_as_of_spine_routing.py`: **22 passed in 1.50 s** (exec `f3b04b`).
The checks cover cutoff-before-top-k refill, all summary channels, same-day
retention, older evidence in mixed sources and leaves, partial attachments,
exact raw bytes, future plans in older statements, unchanged packets when all
dates are eligible, zero raw loads for empty eligibility, pre-budget backfill,
calendar boundaries, ambiguity, foreign descriptors, and query/vector binding.

The real development50 driver completed in session **5805**, exit 0. Run the
same diagnostic on a fresh root using the per-namespace CLI. The archived
driver's valid invocation is:

```powershell
.\.pixi\envs\dev\python.exe -X utf8 -u -c "import runpy; runpy.run_path('eval_results/full1m-spine-as-of-development50-20260910-r1/run.py', run_name='__main__')"
```

That driver intentionally rejects existing namespace audits. Its first direct
script-path invocation failed before execution because the repository root was
absent from Python's import path; the command above completed all five child
processes. No provider request was involved in that invocation failure.

The frozen full100 r3 experiment is unchanged: seven complete memories and
350 prepared requests, with no semantic-seed answer or judge calls. At
**16:17:03 UTC**, raw session **42892** was still ingesting offset 070, with
**576 response files / 580 request files** of 838. Completion scheduler
**12213** remains live and waits for that whole namespace before compiling it.
Both handles were polled successfully, and their original PID/creation-time
identities were rechecked. Neither process was restarted. The same inspection
revalidated all 350 prepared requests and all five date-audit implementation
bindings (exec `fa0fe5`). All model/compiler files bound by their existing
preflights remain unchanged.

Continue observing those two live handles. The new date candidate is separate
from their frozen comparison and requires its own fresh answer and matched
latency evaluation. Full100 still requires ten complete memories, fresh model
readiness, an idle worktree for serving measurements, and sealing all answers
before judging. Historical 95/100 remains the heavier cumulative-validation
result described in Logs 102 and 132. Confirmation remains unopened by this
continuation, with Log 102's historical exposure qualification still applying.
