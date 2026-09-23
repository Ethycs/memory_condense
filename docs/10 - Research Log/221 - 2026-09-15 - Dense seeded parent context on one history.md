# Dense seeded parent context on one history

**Status:** RUN COMPLETE — 84/100, up from 81/100; warm median 3.985 s; 95% target unmet.
**Date:** 2026-09-15.
**Scope:** The unchanged 100-question set and the existing 1,098,417-token memory.
**Depends on:** [Research Log 219](219%20-%202026-09-15%20-%20Complete%20reader%20on%20the%20locked%20single%20history%20100%20questions.md) and [Research Log 220](220%20-%202026-09-15%20-%20Prediction%20blind%20quality%20audit%20of%20the%20100%20question%20set.md).

## Change and rationale

The completed v6 reader run scored 81/100 with the old 1,024-token packet. Ten
misses lacked some or all recorded reference support. Saved routes exposed weak
keyword matches taking the first two seed positions and directing context
expansion into unrelated conversations. Other relevant turns were selected but
did not fit the raw packet.

The next policy uses the existing router and stored attention-derived hierarchy:

| Setting | Previous | Current trial |
| --- | ---: | ---: |
| Reserved lexical prefix | 2 | 0 |
| Ancestor hops | 1 | 2 |
| Context atom limit | 8 | 16 |
| Raw context token cap | 1,024 | 2,048 |
| Direct candidate limit | 32 | 32 |
| Context seed limit | 4 | 4 |
| Protected direct prefix | 0 | 0 |
| Raw span limit | 128 | 128 |

Semantic summary matches therefore supply the seeds. Up to two ancestor levels
can provide later user statements in the same stored chunk, and a larger packet
can retain more of that evidence. Qwen processing remains confined to summaries
at ingestion; the query uses a fresh BGE embedding and existing topology, with
zero raw reads during routing and no query-time Qwen pass. Exact hydration still
owns source reads and token limits. No hierarchy or embedding is recompiled.

The v6 reader, Terra answer model, Sol grading policy, questions and references
remain unchanged. This tests a combined routing/packing policy, not separate
causal effects of each setting. It is development on an exposed question set.
The original 71/100 and v6 81/100 results remain intact, including documented
question-quality defects. No question is removed or regraded to raise the score.

## Implementation and checks

`tools/native_spine_context_policy.py` accepts bounded numeric retrieval settings
only and supplies policy-aware raw reconstruction. It adds no question-specific
source rules. `tools/evaluate_native_spine_context100.py` adapts the verified
one-history reader harness to a sealed policy artifact, so later configuration
trials do not require rewriting the harness. Each trial loads one namespace once,
prepares all prompts, performs fresh retrieval inside timed candidate calls, and
uses alternating identical-prompt API controls. All 200 streams must be sealed
before references are opened for the 100 logical judgments.

Sixteen checks passed in 2.80 seconds. They cover the new policy's live summary
query, exclusion of future raw occurrences, bounded numeric settings, rejection
of question-specific policy keys, exact Unicode source reconstruction, changed
raw-text rejection, and existing context and response-journal behavior.

```powershell
.\.pixi\envs\dev\python.exe -X utf8 -m pytest tests/test_native_spine_context_policy.py tests/test_native_spine_context_routing.py tests/test_native_spine_reader100.py -q --basetemp=.tmp/pytest-context100-20260915-r1
```

## Completed worker

| Item | Binding |
| --- | --- |
| Policy | `eval_results/native-spine-context-policies/dense-parent-2048-v1.json` |
| Policy SHA-256 | `b25c34b8ca304c1b799618722c2d1e000f33566ada6914b0c918db115a797d97` |
| Run root | `eval_results/native-spine-context100-dense2048-20260915-r1` |
| Log | `eval_results/native-spine-context100-dense2048-20260915-r1.log` |
| Terminal exec session | `49208`, exit 0 |
| Worker | PID `37624`, creation time `1789514219.5462208` |
| Locked questions SHA-256 | `76492c4fbc3b142d803eff6bdcb86ac456bb68119431f7da5c9b20d335751bc1` |
| Preflight SHA-256 | `a1393c9b1f79fe0edf15191d227192975ffbb0be487998347ff962ad55cd2905` |

The already released command, for identification rather than duplicate execution:

```powershell
.\.pixi\envs\dev\python.exe -X utf8 -m tools.evaluate_native_spine_context100 run --root eval_results/native-spine-context100-dense2048-20260915-r1 --policy eval_results/native-spine-context-policies/dense-parent-2048-v1.json --enable-provider
```

## Completed score and timing

All 200 answer streams and 100 judgments completed. Every answer ended normally.
The judge replay reproduced the report with 100 authenticated cache hits and no
new model calls. Independent reads of all response files reproduced the timing
distributions and stop counts.

| Measurement | Memory candidate | Matched API control |
| --- | ---: | ---: |
| Accuracy under unchanged grading | 84/100 | Not separately scored |
| Warm median total | 3.985 s | 3.869 s |
| p95 total | 7.328 s | 6.246 s |
| Answers below five seconds | 80/100 | 75/100 |

Candidate median packet preparation was 0.185 seconds. The 37.253-second cold
resident load is excluded from warm latency. The median total is approximately
1.030 times the matched API median. This meets the warm median requirement but
establishes no five-second tail guarantee. Accuracy remains below 95/100.

Compared with the unchanged-reader 1,024-token run, the new policy gains eight
questions (11, 15, 17, 28, 33, 61, 89 and 99) and loses five (9, 14, 35, 62 and 87).
The observed net gain is three points; it is not a claim of statistical superiority
or generalization. The 71/100 and 81/100 baseline artifacts are preserved.

The raw audit reconstructed all 100 memory packets and verified 2,656 raw spans
against the original source bytes, roles, occurrences and dates. A post-answer
support audit finds every recorded reference quote in 97 packets, some in two,
and none in one. It merges adjacent intervals from the same source turn before
checking exact quotes. Reference support coverage is not semantic answer accuracy.

| Artifact in the run root | SHA-256 |
| --- | --- |
| `joint-report.json` | `d06d67b5f15e009411389cec31d0adaa43f5c4194a91de376ff85388ecac14e4` |
| `raw-audit.json` | `7117e6fa6b71779d268c0a9ec663ad8f3384554ff59caff913cfb6e2ab7c83d8` |
| `support-coverage.json` | `6bcb47c48bc36d3611a8f7e893e4c7819ebd7553132c083013adcc783945292f` |
| `comparison-audit.json` | `66fae386f91c837b1529d13e35c06bc043a6ae7140c75d608d564b4ef421db19` |

Replay judging or raw verification without provider calls:

```powershell
.\.pixi\envs\dev\python.exe -X utf8 -m tools.evaluate_native_spine_context100 judge --root eval_results/native-spine-context100-dense2048-20260915-r1
.\.pixi\envs\dev\python.exe -X utf8 -m tools.evaluate_native_spine_context100 audit --root eval_results/native-spine-context100-dense2048-20260915-r1
```

## Remaining failures and next action

Thirteen of the sixteen misses have all recorded support quotes. Several still
omit relevant details (for example, construction today, the budgeting guideline,
or the Mayan creation story). Other cases expose grading defects, including the
unchanged rejection of `B` against `B, not A`, and reference-only rejection of
additional apps despite explicit user statements in the served memory. These
observations do not revise the recorded score.

Only ordinals 35, 53 and 93 lack full recorded support. Item 35 asks what the user
suggested asking someone to help create; its Telegram-group source was lost when
the lexical reserve was removed. Items 53 and 93 are the two independently
documented question ambiguities. Do not add question-specific retrieval rules.

The next work should focus on complete, source-faithful reading and evaluation
validity while preserving the current measured result. Keeping useful lexical
candidates without allowing them to dominate hierarchy seeds is also a concrete
retrieval repair. Avoid increasing corpus scope or repeatedly rebuilding memory.
The historical grouped-renderer/v4 combination in Log 171 was rejected; it does
not prove that grouping alone helps and should not be promoted on assumption.
An apparent threshold pass still requires source-correctness and lifecycle review.
Never resume the stopped 100-history controllers for this task.
