# Eight direct matches through application memory

**Status:** COMPLETE — 93/100; warm median 4.298 s; 95% target unmet.
**Date:** 2026-09-15.
**Scope:** One persisted 1,098,417-token application memory and the unchanged 100 questions.
**Depends on:** [Research Log 226](226%20-%202026-09-15%20-%20Narrower%20retrieval%20packet%20coverage.md).

The coverage diagnostic retained all recorded support on the same 97 questions
at direct-match limits 32, 16 and 8. Width 8 reduced median conversations from 14
to 4. This run tests the hypothesis that reduced cross-conversation material helps
the answer model avoid omissions and attribution errors. Coverage is not accuracy.

Only `max_direct` changes from 32 to 8. The 2,048-token cap, two-level parent
expansion, v7 reader, conversation renderer, answer model, 256-token output cap,
questions, references and grading remain unchanged. The application memory is
reopened read-only; no history ingestion, summary regeneration or Qwen call occurs.

`tools/evaluate_native_spine_application100.py` reuses the existing lifecycle,
retrieval, renderer, streaming journal and grading components. It accepts sealed
numeric retrieval policies while requiring the same reader and question population
as the 87/100 baseline. It validates that each prompt contains exactly the sealed
rendering and unchanged question, with an identical prompt for its API control.

Fifteen focused checks passed in 2.70 seconds. They exercise live width-8 routing,
date exclusion, both alternating arm orders, question/evidence/prompt tampering,
and refusal to grade incomplete answers. The first test run had a malformed date
in the synthetic fixture; correcting its format resolved the failures without
changing production parsing. A provider-free check also validated all 100 real
candidate/control prompt pairs against the completed width diagnostic.

```powershell
.\.pixi\envs\dev\python.exe -X utf8 -m pytest tests/test_native_spine_application100.py tests/test_native_spine_application_reader100.py tests/test_native_spine_application_eval_gate.py -q --basetemp=.tmp/pytest-native-application100-20260915-r2
```

| Item | Binding |
| --- | --- |
| Root | `eval_results/native-spine-app-direct8-20260915-r1` |
| Log | `eval_results/native-spine-app-direct8-20260915-r1.log` |
| Exec session | `30151`, terminal exit 0 |
| Worker | PID `57444`, creation time `1789521658.3316112` |
| Preflight SHA-256 | `f4f8a44addb32c156d7b853b18dd2f46b7ffeeb5cedc103c270b4e1a929dabca` |
| Retrieval policy SHA-256 | `114516284f503fb8ba489b757f8a5ef2283b9e2112b3a25ec41c3a2a5e480e19` |
| Reader policy SHA-256 | `e736763ec6607070bb0c8bfc819589c7e014e8429affde40414750a06c9d6beb` |
| Questions SHA-256 | `76492c4fbc3b142d803eff6bdcb86ac456bb68119431f7da5c9b20d335751bc1` |
| Application reopen verification SHA-256 | `ca721112762da52024df005868c3b092b8f684c5d37d2e9f99af62497c88d2e0` |
| Report SHA-256 | `463e67ec477be916ef8bd8b533f1af71817140c2f99babf0ab4dac3e8d8fd841` |
| Raw audit SHA-256 | `69a208c05b72b306435434fec77ec8bba613debb7413eee92e7a22845ee1ee04` |
| Independent comparison SHA-256 | `2a2210b7200cc108b040a741c06445eb126ec93b1088d6e9618f4305e51237a9` |
| Failure diagnosis SHA-256 | `f29ba6bace2f544e5aeef36c5c108fcb64646bbeab730e3fb9984ea3d265a3f0` |

The completed command is recorded for identification; do not duplicate it:

```powershell
.\.pixi\envs\dev\python.exe -X utf8 -u -m tools.evaluate_native_spine_application100 run --root eval_results/native-spine-app-direct8-20260915-r1 --policy eval_results/native-spine-context-policies/dense-parent-2048-direct8-v1.json --reader-policy eval_results/native-spine-reader-policies/user-coverage-v7.json --enable-provider
```

The schedule contains 100 memory answers and 100 matched API controls. Every
memory answer performs fresh application retrieval inside its timer. All answers
must seal before references open for 100 judgments. The final audit reconstructs
every served span and conversation rendering against the original raw sources.
Warm latency excludes separately measured ingestion and cold application load.

An additional provider-free comparison confirmed that all 100 freshly retrieved
application packets exactly match the width-8 diagnostic's routing, hydration
and rendering. Its receipt, `diagnostic-packet-comparison.json`, has SHA-256
`30e8d8f311cb6518ca0855180019096b0b451316ecd197df309c16fe9f73247d`.

## Completed score and timing

All 200 answer streams stopped normally. All 100 judgments completed. A
provider-disabled replay reproduced the report with 100 authenticated cache hits
and zero new calls. An independent comparison recomputed every latency statistic
from the response files. All 100 packets and 1,430 exact raw spans passed audit.

| Measurement | Application memory | Matched API |
| --- | ---: | ---: |
| Accuracy under unchanged grading | 93/100 | Not separately scored |
| Warm median total | 4.298 s | 4.251 s |
| p95 total | 6.741 s | 7.088 s |
| Answers below five seconds | 73/100 | 74/100 |
| Median packet preparation | 0.168 s | Less than 0.001 s |

Cold application/model load was 28.499 seconds, excluded from warm latency.
The median memory/API ratio is approximately 1.011. The accepted median latency
threshold is met, while the 95% accuracy target is still unmet.

Gains against the 87/100 width-32 run are ordinals **0, 14, 48, 69, 81, 89, 95**;
the sole regression is **19**. Narrowing the packet recovered several omissions
without changing the reader or answer model. This supports reducing irrelevant
context, but one exposed-set run does not establish a reproducible six-point
population improvement.

## Remaining misses and next comparison

Misses are **19, 26, 35, 38, 53, 82, 93**. Manual source inspection distinguishes:

- **19 and 82:** clear reader omissions despite explicit served user text. The
  answers lose the possibility that personal experience adds acting depth and
  the preference for thoughtful real-world issues/current-events commentary.
- **35:** missing target evidence. The vague question refers to an unnamed
  person and asks what to create; the expected Telegram-group statement is not
  in the packet. A better reader cannot recover absent evidence reliably.
- **26 and 38:** the grader rejects supported detail. A real user request asks
  for Indigenous cultural events/festivals in the Northwest Territories, and
  the same game conversation explicitly calls it educational. The reference
  omits these details. The question in 26 does not specify a conversation.
- **53 and 93:** previously confirmed ambiguity among actual concerts/festivals
  and vintage collections. There are multiple valid source conversations.

The TV-drama reference also includes unrequested show/character details, but
this run's answer independently omits real preferences, so it remains a reader
failure. These categories do not revise any reference, verdict or accuracy score.

**Correction to the earlier diagnosis:** the educational-game detail was also
present as user text in the 87/100 run. Its earlier classification as a model
invention was wrong. The immutable result is unchanged; the diagnosis is corrected.

A model-only comparison on these identical packets is now justified to test the
remaining reader omissions. Keep Qwen attention on summaries and exact raw-section
hydration; raw-text pruning has not been shown necessary by this result. Preserve
the current candidate and all original scores. Question/reference repairs require
a separately labeled evaluation; they must not silently raise this score.

This is an exposed generated-question development set over real transcripts,
not an official LongMemEval or generalization result. The 95% target remains unmet.
