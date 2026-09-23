# Answer model comparison on identical user spine packets

**Status:** COMPLETE — 93/100, tied with Terra; warm median 4.615 s; target unmet.
**Date:** 2026-09-15.
**Scope:** The same persisted 1,098,417-token history and unchanged 100 questions.
**Depends on:** [Research Log 227](227%20-%202026-09-15%20-%20Eight%20direct%20matches%20through%20application%20memory.md).

The width-8 user-spine candidate scored 93/100 with Terra. Source inspection
found two clear reader omissions despite explicit served user statements. This
comparison tests whether a different answer model improves the complete result
while retaining near-API latency. Model quality is measured, not assumed.

The gateway's model inventory lists `codex_sdk/gpt-5.6-sol`, which already works
for question authoring and judging. This run changes only the answer model from
Terra to Sol. The model alias and inventory are sealed; this identifies the
requested gateway route, not an independently inspected underlying checkpoint.

The user-spine summaries, stored Qwen attention topology, parent expansion,
eight direct matches, 2,048-token cap, v7 reader, exact raw evidence and conversation
layout remain identical. Qwen still receives only summaries during compilation;
there is no query-time Qwen pass or attention pruning within raw text. The normal
application ingestion and separate-process reopen were verified previously. This
run reopens that existing memory read-only and performs fresh timed retrieval for
each memory answer. It does not ingest another history or regenerate summaries.

## Model and evidence validation

`tools/evaluate_native_spine_application_model100.py` reuses the existing
application, retrieval, reader, streaming and grading components. Its model-aware
response journal validates the selected model instead of the earlier hard-coded
Terra route. Every response must retain its sealed request, prompt, prediction
hash, 256-token output limit, raw hydration, routing and rendering.

All 200 preflight prompts and all 100 freshly retrieved evidence packets must
equal the completed 93/100 baseline. Timed retrieval must reproduce them again.
Sol answers and matched direct-Sol controls alternate order across questions.
The controls use identical prompts; their time excludes memory preparation.

Nineteen focused tests passed in 4.09 seconds. They check selected-model binding,
candidate/control evidence, altered predictions/rendering/output limits,
unacknowledged streams, journal gaps, missing answers, admitted model inventory,
and the existing application and population gates. The raw-answer model boundary
rejects Qwen. No application or retrieval code changed.

```powershell
.\.pixi\envs\dev\python.exe -X utf8 -m pytest tests/test_native_spine_application_model100.py tests/test_native_spine_application100.py tests/test_native_spine_application_eval_gate.py -q --basetemp=.tmp/pytest-native-model100-20260915-r1
```

## Released execution

| Item | Binding |
| --- | --- |
| Root | `eval_results/native-spine-app-sol100-20260915-r1` |
| Log | `eval_results/native-spine-app-sol100-20260915-r1.log` |
| Exec session | `18164`, terminal exit 0 |
| Worker | PID `56612`, creation time `1789523403.2723656` |
| Preflight SHA-256 | `df7b54891ef4f64ce0ffbdeded107bcd7676e22465b7c2ef06e9a57981ab0882` |
| Answer model | `codex_sdk/gpt-5.6-sol` |
| Model inventory SHA-256 | `d89a805de41673bfa49df7a46d04c17370bb39cebace674e91239e9f27cbbfaf` |
| Retrieval policy SHA-256 | `114516284f503fb8ba489b757f8a5ef2283b9e2112b3a25ec41c3a2a5e480e19` |
| Reader policy SHA-256 | `e736763ec6607070bb0c8bfc819589c7e014e8429affde40414750a06c9d6beb` |
| Questions SHA-256 | `76492c4fbc3b142d803eff6bdcb86ac456bb68119431f7da5c9b20d335751bc1` |
| Baseline report SHA-256 | `463e67ec477be916ef8bd8b533f1af71817140c2f99babf0ab4dac3e8d8fd841` |
| Report SHA-256 | `aa451f1f895ca7902db4dfd0ca12c4ad11da91a937156de72ca3adb970ee2d8e` |
| Raw audit SHA-256 | `c315b1afbcdc75daeb2f16f7fc59352acce61bf58ceae87137dacae644d6efdc` |
| Independent comparison SHA-256 | `66ebc79b8cabe6968dae8c7172785ff1a153fd7943dbf2356bb1e74b09d6e404` |

The completed command is recorded for identification; do not duplicate it:

```powershell
.\.pixi\envs\dev\python.exe -X utf8 -u -m tools.evaluate_native_spine_application_model100 run --root eval_results/native-spine-app-sol100-20260915-r1 --policy eval_results/native-spine-context-policies/dense-parent-2048-direct8-v1.json --reader-policy eval_results/native-spine-reader-policies/user-coverage-v7.json --model codex_sdk/gpt-5.6-sol --enable-provider
```

There are 100 memory answers, 100 matched API controls and then 100 judgments.
References open only after every answer is sealed. The judge model and prompt
remain unchanged, so Sol is both this candidate's answer model and the grader.
All known question/reference defects remain visible; no verdict is silently
corrected. The final source audit reconstructs every served raw span and prompt.

## Completed result

All 200 streams ended normally and reported the requested Sol alias. All 100
judgments completed, and the provider-disabled replay reproduced the same report
with 100 authenticated cache hits and zero new calls. Independent recomputation
matched every reported timing statistic. All 100 packets and 1,430 exact raw spans
passed the original-source audit.

| Measurement | Application memory | Matched direct Sol |
| --- | ---: | ---: |
| Accuracy under unchanged grading | 93/100 | Not separately scored |
| Warm median total | 4.615 s | 4.383 s |
| p95 total | 8.500 s | 6.852 s |
| Answers below five seconds | 64/100 | 73/100 |
| Median packet preparation | 0.286 s | Less than 0.001 s |

Cold model/application load was 29.764 seconds, excluded from warm timing.
The ratio of warm medians was approximately 1.053. The score ties Terra's 93/100;
this experiment supplies no total-score reason to promote Sol. Terra's prior
measured median was 4.298 seconds, though separate runs also contain gateway and
local timing variability. The 95% accuracy target remains unmet.

Sol gains ordinals **19, 26, 38**, and loses **95, 98, 99**. Misses are
**35, 53, 82, 93, 95, 98, 99**. It recovers the acting-depth omission in 19 and
answers 26/38 with fewer details that the incomplete references had penalized.
Its remaining failures mix retrieval, reading and evaluation issues:

- **35:** the unchanged packet still lacks the Telegram-group statement.
- **53 and 93:** known ambiguity among real concert/festival and collection facts.
- **82:** Sol now gives the substantive drama requirements, but the grader rejects
  omission of unrequested show/character names. This differs from Terra's actual
  omission of stated preferences on the same question.
- **95:** the answer includes the gold preferences and Tribeca; the grader rejects
  additional French/international-cinema interests explicitly present in user text.
- **98:** the answer identifies the expected title, but calls it a song and corrects
  the question's album premise. Served assistant context repeatedly calls it a song;
  the user says they will listen to the title without specifying its type. This
  exposes a source/premise inconsistency rather than a wrong-title retrieval.
- **99:** a clear reader omission of software/equipment access, despite all three
  target user turns being present.

No question, reference, verdict or score has changed. These source checks diagnose
the result; they do not create an adjusted accuracy claim.

## Next routing check

The missing Telegram fact is already in the persisted summary index, dated before
the question. Its summary says that the user wants Jo to ask Vladimir whether he
wants to help create a Telegram group. Both earlier hybrid-routing runs answered
this question correctly; it became a miss with the dense-only policy.

The next provider-free check restores one lexical summary match within the same
eight direct candidates. It must inspect coverage for all 100 questions, seal
packets before opening references, and independently reconstruct exact raw text.
This is conventional summary routing, with no raw-text attention or new ingestion.
The existing answer runner can test its sealed numeric policy without another
model or reader change if coverage warrants a full run. See Research Log 229.

This exposed development set is not an official LongMemEval or generalization
result. The verified best total score remains 93/100, and the goal stays active.
