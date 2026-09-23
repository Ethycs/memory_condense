# Complete user statement reader through application memory

**Status:** COMPLETE — 87/100; warm median 4.465 s; 95% target unmet.
**Date:** 2026-09-15.
**Scope:** The existing persisted 1,098,417-token application memory and unchanged 100 questions.
**Depends on:** [Research Log 224](224%20-%202026-09-15%20-%20Conversation%20ordered%20answers%20through%20persisted%20application%20memory.md).

## Change and rationale

The completed grouped v6 reader scored 85/100. Several misses omitted explicit
qualifications and follow-up points despite their presence in the packet. Another
copied quantities from an assistant recommendation into the user's history.

The v7 policy makes the reading procedure explicit: match the full situation,
inspect all relevant user turns, cover distinct requested points, preserve
tentative considerations and rejected alternatives, keep the meaning of recalled
passages, and check speaker attribution for every claimed user fact. It asks for
the final answer only. Its 800-token system prompt contains no test-specific
entities, answers, source IDs, ordinals or reference text.

The retrieval policy, selected raw spans, grouped presentation, questions,
references, Terra answer model, 256-token output limit and Sol grading are
unchanged. The preflight independently compares all 200 candidate/control prompts
with the previous run and permits only the sealed system-instruction replacement.
No new history ingestion or Qwen compilation occurs. Cached summaries remain the
only inputs to the stored Qwen hierarchy; answers hydrate original raw evidence.

## Implementation and checks

`spine_reader_policy_v7.py` supplies the new policy and a copy-on-write prompt
adapter. `evaluate_native_spine_application_reader100.py` reuses the established
application lifecycle, raw renderer, response-journal validation and grading
implementations. Reader instructions are a separately sealed artifact so later
instruction comparisons need not rewrite the runner. The prior implementations,
scores and references are preserved.

Ten focused checks pass in 2.54 seconds. They verify exact live retrieval and
rendering preservation, future-occurrence exclusion, unchanged questions and
baseline messages, policy bounds, rejection of extra reference inputs, and the
existing response/lifecycle gates. A provider-free check adapted all 200 saved
baseline prompts and confirmed their context/question messages were unchanged.
These checks establish isolation, not answer quality.

```powershell
.\.pixi\envs\dev\python.exe -X utf8 -m pytest tests/test_native_spine_application_reader100.py tests/test_native_spine_reader100.py tests/test_native_spine_application_eval_gate.py -q --basetemp=.tmp/pytest-native-reader-v7-20260915-r1
```

## Released worker

| Item | Binding |
| --- | --- |
| Root | `eval_results/native-spine-app-reader100-v7-20260915-r1` |
| Log | `eval_results/native-spine-app-reader100-v7-20260915-r1.log` |
| Exec session | `81985`, terminal exit 0 |
| Worker | PID `43736`, creation time `1789519129.1858375` |
| Preflight SHA-256 | `03c280c5304804d04ac132a528e06fa2ae65ce36f397f5cbd001436c2f1deb7d` |
| Reader artifact | `eval_results/native-spine-reader-policies/user-coverage-v7.json` |
| Reader SHA-256 | `e736763ec6607070bb0c8bfc819589c7e014e8429affde40414750a06c9d6beb` |
| Retrieval policy SHA-256 | `b25c34b8ca304c1b799618722c2d1e000f33566ada6914b0c918db115a797d97` |
| Questions SHA-256 | `76492c4fbc3b142d803eff6bdcb86ac456bb68119431f7da5c9b20d335751bc1` |
| Report SHA-256 | `831ff4b089106a83d51a9dad687226b6ea2e2c7b1f7a40bad6409ede318fecb3` |
| Raw audit SHA-256 | `34a72e2688cd39abce22281e62aa8430f8c25a643ab805e5ff8b41e33f5da156` |
| Independent comparison SHA-256 | `c1c45df941bc2e2be6de442645f800a3d2fdeb2e821eb6a1caf4e9c1ceb580c5` |

The completed command is recorded for identification, not duplicate execution:

```powershell
.\.pixi\envs\dev\python.exe -X utf8 -u -m tools.evaluate_native_spine_application_reader100 run --root eval_results/native-spine-app-reader100-v7-20260915-r1 --policy eval_results/native-spine-context-policies/dense-parent-2048-v1.json --reader-policy eval_results/native-spine-reader-policies/user-coverage-v7.json --enable-provider
```

There are 100 fresh memory answers, 100 matched API controls, then 100 logical
judgments. References open only after all answers are sealed. The final raw audit
reconstructs all served text and rendering against original sources. Warm timing
includes application retrieval and rendering but excludes cold load/ingestion.
This is development on exposed generated questions over real transcripts, not
an official LongMemEval or generalization score.

## Completed score and validation

All 200 answer streams stopped normally. All 100 judgments completed, and a
provider-disabled replay reproduced the report with 100 authenticated cache hits
and zero new calls. The independent comparison recomputed every latency
distribution from response files. All 100 packets and 2,656 original raw spans
passed reconstruction, including unchanged rendering and the new system prompt.

| Measurement | Application memory | Matched API |
| --- | ---: | ---: |
| Accuracy under unchanged grading | 87/100 | Not separately scored |
| Warm median total | 4.465 s | 3.962 s |
| p95 total | 7.015 s | 6.940 s |
| Answers below five seconds | 71/100 | 79/100 |
| Median packet preparation | 0.304 s | Less than 0.001 s |

Cold application/model load was 25.760 seconds, excluded from warm latency.
The median memory/API ratio is approximately 1.127. The user accepted median
latency below five seconds; answer accuracy remains below the 95% target.

Gains against grouped v6 are ordinals **1, 5, 6, 19, 65, 96**; regressions are
**26, 38, 48, 69**. The policy recovers the Subway speaker-attribution error and
several missing qualifications, but the net gain is only two points. Misses are
**0, 14, 26, 35, 38, 48, 53, 69, 81, 82, 89, 93, 95**.

Several true reader errors remain: omitted construction/creation-story details,
the wrong commentary preference, and an abbreviated
opening line that loses its central contrast. Evaluation limitations also remain:
`B.` is penalized for not adding `not A`; the complete TV-drama requirements are
penalized for omitting unrequested show/character names; a real later request for
post-revolution conversations is called unsupported; and the concert/collection
questions remain ambiguous. Original scores, references and questions are unchanged.

Later source inspection in Research Log 227 corrects the initial game-attribute
diagnosis: the user explicitly called the game educational in this run's served
packet. The grader rejected supported detail; it was not a model invention.

## System-message diagnostic and next direction

A separate six-call control tested three short exact-output instructions in the
system role and their equivalent user-role controls. All six succeeded and ended
normally. This rules out system messages being dropped wholesale in those
controls; it does not prove compliance with a long reader policy.

The control used no history, benchmark questions or references. Root:
`eval_results/native-spine-system-message-control-20260915-r1`; report SHA-256:
`cca0e8b672111aedf8fc2d4182b9cb4097f72bd4b3d7db72a2ad8d64ce5c9cd9`.
Its exec session `31048` exited 0. No additional answer-accuracy score is implied.

Current packets contain 6–26 conversations, median 14. Before another instruction
revision or a different answer model, the next check reduces this cross-conversation
material while measuring support retention on all 100 questions. Research Log 226
records that completed provider-free diagnostic. Any resulting candidate still
requires a fresh full answer/latency run through the persisted application.
