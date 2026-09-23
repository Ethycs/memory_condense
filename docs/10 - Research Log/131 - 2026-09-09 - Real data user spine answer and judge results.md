# Real data user spine answer and judge results

**Status**: COMPLETE, SEALED AND REPLAYED — DEVELOPMENT PILOT; NO ROUTER PROMOTED
**Date**: 2026-09-09
**Applies to**: `perf/durable-ingest-pipeline`, `.worktrees/ingest-speed`
**Depends on**: [Research Log 130](130%20-%202026-09-09%20-%20Real%20data%20user%20spine%20routing%20evaluation.md)

## Result

The prepared real-data answer evaluation is complete. All five variants scored
**7/7 on seven source-derived diagnostic questions and 0/1 on the separate
benchmark trip-chronology question**. Attention-based Qwen routing has not
demonstrated an answer-accuracy improvement over the shared-hierarchy BM25
control. No default retrieval path changes and no candidate is promoted.

| Variant | Diagnostic answers | Benchmark chronology | Qwen routing input tokens, eight queries | Mean Qwen provider seconds/query |
| --- | ---: | ---: | ---: | ---: |
| BM25, both summary channels | 7/7 | 0/1 | 0 | — |
| Qwen narrow | 7/7 | 0/1 | 51,931 | 8.71 |
| Qwen exploration beam 8 | 7/7 | 0/1 | 63,503 | 10.65 |
| BM25, user channel only | 7/7 | 0/1 | 0 | — |
| Qwen narrow, user channel only | 7/7 | 0/1 | 33,610 | 6.61 |

User-only Qwen routing used 35.3% fewer input token proxies and 24.2% less
observed provider time than narrow Qwen with both channels. This is a measured
efficiency result on this pilot, without an accuracy gain. Timing sums recorded
provider-call durations; it is not end-to-end latency and does not include
compilation, local routing work, answering or judging. The observations are
single runs, with no confidence interval. A zero Qwen call count is not a claim
of zero BM25 runtime.

## Matched scope and evidence boundary

The population remains the same 39 turns in three complete real conversations
selected from the existing r9 q86 candidate packet. It contains 18 user turns
and 8,855 raw token proxies. The seven diagnostic questions were derived from
those sources before initial routing; their references were later inspected
for the separately sealed evidence-coverage audit. The user-only projection
was an explicitly labeled development follow-up. None is untouched validation.
The one original benchmark question is reported separately from those probes.
This result cannot be compared as if it were the earlier reduced30 or full100.

The first three variants share identical summary strings and hierarchy. The
two projection variants change only the summary channel exposed for routing;
they preserve topology, source membership, raw spans, questions and budgets.
All variants allow three sections, 4,096 raw-context tokens, 128 raw spans and
a 5,500-token answer prompt. Actual contexts vary with selection. Exact raw
content, roles and transcript dates are reconstructed after routing; generated
summaries do not appear in answer evidence. No hydration diagnostics occurred.

Terra generated all answers using the same QA instructions and a 256-token
output cap. Sol judged the frozen predictions against the same per-question
references using the common binary judge prompt and a 32-token output cap.
Benchmark references were joined only after both real answer populations
sealed. The loader reconstructed the locked validation population identity
`9b8ad9337cfece1306358d0e03682a977f1b289a14b6ff7bfe40c90e6e2cb246`.
The seven development references had already been inspected after answer inputs
sealed; none was added to those frozen input messages.

An additional audit checked all **133 Qwen routing requests**. Their candidate
text exactly matches the corresponding frozen hierarchy summaries, with no
raw-locator or raw-content fields. The query population is unchanged. Raw
summary compilation and answer generation use Terra, not Qwen.

For precision clarity, the local attention pass used **FP16 prefix weights and
forward computation**, with FP32 softmax and pooled readout calculations. It
reads layer 6's QK/OV signals after executing the preceding five full transformer
blocks, including their MLPs; layer 6's MLP is skipped. It is not an FP32 model
made solely of attention heads. The full gateway Qwen model separately performs
summary merging and routing. Gateway weight precision was not measured here.

## Why chronology still fails

Four variants answered `I don't know`. The wider Qwen router answered
`Yosemite, Big Sur and Monterey, Eastern Sierra`; Sol marked it incorrect.
The reference order is Muir Woods, Big Sur/Monterey, then Yosemite. The wider
packet lacks the Muir Woods evidence and includes planning material, so its
answer promotes a planned trip into the completed-trip sequence.

The sealed plans explain the failure before generation:

- Both BM25 variants selected only the two March user exchanges at ordinals
  7 and 9. The only matched query term was `trips`; those summaries discuss
  backpack requests for upcoming trips. Other relevant summaries use singular
  `trip`. The conventional control therefore has an observable lexical
  limitation; this does not rule out stemming, dense retrieval or other
  conventional approaches.
- Both narrow Qwen variants selected user turns 27, 18 and 9. They covered all
  three source conversations, but April turn 18 is about a planned July trip
  and backpack suitability. It supplies no Big Sur/Monterey event evidence.
  Source coverage is not event coverage.
- The wider Qwen variant selected 29, 18 and 14. Increasing exploration did not
  yield a complete set of completed-trip assertions. The May source also has
  conflicting "got back today" and "started today" statements. Mention dates
  cannot safely resolve that conflict automatically.

Exact hydration is working; the evidence set is incomplete. The seven simpler
entity, preference and observation questions demonstrate useful retrieval and
answering but do not test reliable multi-event coverage. The next meaningful
experiment would use query-derived event requirements and explicit evidence
coverage, retaining event status and contradictions, on additional independent
question families. These trip names and turn ordinals must not become routing
rules. Wider exploration alone is not supported as an improvement by this run.

All 21 pilot leaves are single user-led exchanges or preludes. Attention changes
the parent grouping here; this experiment does not isolate a benefit from
attention-selected raw chunk boundaries. It also does not establish that local
FP32 execution would improve routing.

## Execution, replay and artifact identities

The earlier automatic-review rejection is resolved. After the exact remaining
scope had been presented and the user said "keep going," automatic review
accepted both Terra answer launches and both Sol judge launches. The original
packets and limits were preserved. The resumption receipt SHA is
`351dd583fbc8fe1b6195387d25fb9a75f2360508afdaa5c9393c3829088650b0`.

| Phase | Logical rows | Unique live calls | Zero-call replay hits |
| --- | ---: | ---: | ---: |
| Matched three-arm answers | 24 | 22 | 22 |
| User-channel two-arm answers | 16 | 15 | 15 |
| Matched three-arm judgments | 24 | 14 | 14 |
| User-channel two-arm judgments | 16 | 10 | 10 |
| Total | 40 predictions + 40 judgments | 61 | 61 |

Identical prompts deduplicate within a phase and expand back to every logical
case/arm row. Every row contributes to its group's denominator. All completions
finished normally, with zero SDK retries. This resumption added 37 Terra answer
calls and 24 Sol judge calls, with no new Qwen calls. Including the preceding
compilation and routing work recorded in Log 130, there are 289 completed calls;
the eight older r2 request reservations without responses remain a separate
aborted transport attempt, not successful completions.

The final reporting command authenticated all answer and judge journals,
reconstructed all input messages from exact raw spans, checked matching
question/reference populations, hierarchy topology, budgets, case/arm row
alignment and aggregate counts, and reproduced the same prediction and
judgment artifact hashes. It created **zero new provider calls**.

Paths are relative to this worktree's `eval_results/`:

| Artifact | SHA-256 |
| --- | --- |
| `user-spine-real-matched-pilot-20260909-r1/answers.json` | `303c703cd566a6ff55f40695721204eaafdb02961121650349d5e3c75c0782a9` |
| Same root, `judge-preflight.json` | `79b6d33766165c76e9d46ae1c1a81c8ac5fa9523d120e566c9221052a954f88e` |
| Same root, `judgments.json` | `4daa5f530fc89c2f0d024338e75498de5615a7310d302544f9649229dce5f6e7` |
| `user-spine-real-projection-pilot-20260909-r1/answers.json` | `d529be57c0857a24b542b0f7f9617873e3b1616a565412ccc674970d72b1c890` |
| Same root, `judge-preflight.json` | `d4879fafc00fc948fb2601ebace3faa0c9e151d15ca38a01ba50857d6332e383` |
| Same root, `judgments.json` | `7b7a20ca59dd39f1406873d1009edac16284ca70cf82d10c4eba1e924a0500ab` |
| `user-spine-real-matched-pilot-20260909-r1/completed-evaluation.json` | `0110988e09df864a37423ea9d4b07a3396f3c7fb2b338d8f36df14aa7190056c` |
| Same root, `qwen-routing-input-audit.json` | `c9f344e41f581692bc199193b50b58e5ac2f734f5de94f096e6f4f81b805cc4b` |

Run from `.worktrees/ingest-speed` to authenticate and reproduce the completed
report without a provider client:

```powershell
.\.pixi\envs\dev\python.exe -X utf8 -m tools.report_user_spine_real_evaluation --matched-root eval_results/user-spine-real-matched-pilot-20260909-r1 --projection-root eval_results/user-spine-real-projection-pilot-20260909-r1
```

The preceding local integration suite passed **115 tests**, including synthetic
answer/judge lifecycle tests. Those tests validate protocol behavior; the live
results and authenticated replay above supply the actual answer-quality
evidence. The new final report was executed successfully on both real artifact
populations. Broader independent validation remains a prerequisite for promotion.
