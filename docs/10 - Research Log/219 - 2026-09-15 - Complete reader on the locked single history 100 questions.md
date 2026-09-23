# Complete reader on the locked single history 100 questions

**Status:** RUN COMPLETE — 81/100, up from 71/100; warm median 3.695 s; 95% target not met.
**Date:** 2026-09-15.
**Scope:** The same 100 questions and the same ingested 1,098,417-token history.
**Depends on:** [Research Log 218](218%20-%202026-09-15%20-%20Correct%20scope%20to%20100%20questions%20on%20one%20ingested%20history.md).

## Evidence for the change

The completed baseline scored 71/100 with a warm median total of 3.700 seconds.
Nineteen of its 29 misses contained all recorded reference support quotes in the
served packet. Many responses dropped requested details. The reader instructed
the answer model to be as short as possible, even though questions could ask for
several facts, constraints or reasons.

`src/memory_condense/eval/spine_reader_policy_v6.py` replaces that brevity clause
with concise, complete answers covering all requested parts. It preserves v5's
evidence, temporal, action-versus-intent, denial and correction rules. It changes
the final response cue from `Short answer:` to `Answer:`. No example answer,
question identifier, source identifier or reference answer is in the policy.

The new driver, `tools/evaluate_native_spine_reader100.py`, reuses the existing
one-history loader, resident memory, live retrieval, streaming measurement,
journal authentication, judge and raw reconstruction. It loads the memory once.
Before release it verifies that all 100 hydration packets and route receipts are
identical to the baseline. Every timed candidate repeats live retrieval and
compares the resulting packet to its sealed preflight.

Each of the 100 questions has one fresh candidate stream and one fresh API control
with identical prepared messages, alternating order by question: 200 streams,
100 logical judgments, one history. There are no new compilation jobs. Reference
answers are opened only after all 200 answers are sealed. The original question
manifest and judge policy remain unchanged. This is development on questions
whose baseline results have been inspected; it is not a new held-out evaluation.

## Validation and execution

Six focused checks passed in 1.87 seconds: original source-scope checks plus raw
prompt preservation, compatibility with authenticated response journals, refusal
to grade an incomplete run, and rejection of an unexpected reader policy.

```powershell
.\.pixi\envs\dev\python.exe -X utf8 -m pytest tests/test_native_spine_reader100.py tests/test_single_history100_scope.py -q --basetemp=.tmp/pytest-reader100-20260915-r1
```

| Item | Binding |
| --- | --- |
| Run root | `eval_results/native-spine-reader100-v6-20260915-r1` |
| Log | `eval_results/native-spine-reader100-v6-20260915-r1.log` |
| Terminal exec session | `4218`, exit 0 |
| Worker | PID `53592`, creation time `1789512389.4462206` |
| Preflight SHA-256 | `a5109df06ea3445f58f454f1a573e2a1d0c3c01c21de20f8f2db8ac9c470f680` |
| Frozen questions SHA-256 | `76492c4fbc3b142d803eff6bdcb86ac456bb68119431f7da5c9b20d335751bc1` |

The worker completed all 200 streams with normal stops, then 100 judgments and
the raw audit. It logged one history with 1,098,417 actual tokens. No new history
or ingestion was created. Replaying the judge produced the identical report with
100 authenticated cache hits and zero new calls.

The already released command, for identification:

```powershell
.\.pixi\envs\dev\python.exe -X utf8 -m tools.evaluate_native_spine_reader100 run --root eval_results/native-spine-reader100-v6-20260915-r1 --enable-provider
```

## Completed result

| Measurement | Complete reader | Matched API control |
| --- | ---: | ---: |
| Accuracy under unchanged grading | 81/100 | Not separately scored |
| Warm median total | 3.695 s | 3.780 s |
| p95 total | 6.644 s | 6.104 s |
| Answers below five seconds | 86/100 | 82/100 |

The resident cold load took 36.608 seconds, excluded from warm latency. Candidate
preparation had a median of 0.185 seconds. Timing distributions and stop counts
were independently recomputed from all 200 response files. The matched controls
measure the same prompts; the difference in medians does not prove that memory
retrieval itself makes a model call faster.

The reader gained eleven correct answers (ordinals 5, 9, 20, 25, 31, 42, 48, 58,
59, 62 and 87) and lost one (33), for a net gain of ten points. The raw audit
reconstructed all 100 memory packets and verified 1,410 exact source spans.
Retrieval and raw packets were identical to the baseline throughout.

| Artifact | SHA-256 |
| --- | --- |
| `joint-report.json` | `f48607662237708b5a13b2d399b9d4a09427bf8f5ba2b722aadf32da5244542e` |
| `raw-audit.json` | `ed6444e4f70ae3e9f7b7531becfead809e121525456a2975d3cc879e25ab0c5d` |

The 95/100 target remains unmet. Preserve the original 71/100 baseline and this
81/100 result. Nine current misses contain all recorded support quotes, while ten
have incomplete recorded support. Some of the nine also reveal evaluation-item
defects: redundant negation is required for the multiple-choice answer, and some
references demand material beyond the question. A separate prediction-blind audit
of all 100 items completed; it flagged six items and changed no score. See
[Research Log 220](220%20-%202026-09-15%20-%20Prediction%20blind%20quality%20audit%20of%20the%20100%20question%20set.md)
for confirmed ambiguities, model-review limitations and the next retrieval step.

## Remaining retrieval diagnosis

The baseline also has ten misses without all recorded support. Across those
questions, eleven missing support quotes refer to turns absent from the expanded
route and six refer to turns in that route that were not served. These are quote
counts, not independent questions or an accuracy score.

Inspection also found lexical prefix pollution: for the vintage-collection
question, the two reserved lexical seeds matched generic words (`collect`,
`specific`, `details`, `share`) in unrelated weathering and video-advice summaries.
Those seeds also direct parent-context expansion. Investigate this as a generic
summary-routing problem after the isolated reader comparison; it does not justify
question-specific source selection or sending raw content through Qwen.
