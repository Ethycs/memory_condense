# Prediction blind quality audit of the 100 question set

**Status:** DIAGNOSTIC COMPLETE — six model-flagged items; accuracy unchanged at 81/100.
**Date:** 2026-09-15.
**Scope:** All 100 locked questions on the same ingested history, no new answers or ingestion.
**Depends on:** [Research Log 219](219%20-%202026-09-15%20-%20Complete%20reader%20on%20the%20locked%20single%20history%20100%20questions.md).

## Why this audit exists

The reader comparison improved the unchanged score from 71/100 to 81/100. Some
remaining failures reveal incomplete answers or missing evidence; others expose
evaluation problems. The grader, for example, rejected `B` against `B, not A`.
Repairing a serving system to satisfy an invalid requirement can reduce actual
correctness. Diagnose evaluation quality separately from answer performance.

`tools/audit_native_spine_question_quality.py` reviews every locked item, with
candidate predictions withheld. Inputs are the question, unchanged reference,
all user turns from its source body, and the baseline packet's retrieved user
excerpts. All are bound to the existing source bank and completed baseline. No
question or reference enters ingestion, and no score is rewritten.

The Sol gateway is asked to identify unsupported reference claims, requirements
beyond the question, and concrete ambiguity demonstrated by competing supplied
user statements. Every reported issue must quote exact supplied text. The model
may miss defects or overstate them; its output is a diagnostic, not proof that an
item is valid or invalid. The retrieved excerpts are not an exhaustive search
of the entire memory for alternative answers.

## Actual result

All 100 reviews completed in exec session 2745, exit 0. All JSON and source-quote
validations passed. The model flagged six items: one unsupported-reference flag,
two ambiguity flags and three unrequested-requirement flags. These are flags,
not six adjudicated benchmark defects and not an adjusted accuracy numerator.

| Ordinal | Model flag | Review finding |
| ---: | --- | --- |
| 38 | Unsupported reference | Interprets a feasibility question about a game as weaker than an intention. This is debatable: the requested project attribute is directly stated. |
| 53 | Ambiguous question | The memory explicitly describes two different sets of recent concerts and festivals, with no distinguishing event or conversation in the question. Confirmed by inspecting the quoted source text. |
| 82 | Unrequested requirement | Reference adds a selected show and character interest to a question about drama requirements. Other requested requirements are still genuinely missing from the prediction. |
| 85 | Unrequested requirement | Reference includes a research commitment beyond the requested type of business. |
| 93 | Ambiguous question | The memory contains vinyl records, cameras, sci-fi novels and postage stamps, while the question asks for exactly three collections without identifying a conversation. Confirmed by inspecting source quotes. |
| 94 | Unrequested requirement | Reference includes how promotions will be displayed beyond the requested offers. |

The auditor did not flag item 000's redundant `not A`; the directly observed
grading anomaly remains documented. A clean model assessment is not sufficient
to establish evaluation validity. Neither the original 71/100 nor the reader's
81/100 result is changed, and all 100 items remain in their original denominator.

## Reproducible artifacts

Root: `eval_results/native-spine-question-quality-20260915-r1`.
Log: `eval_results/native-spine-question-quality-20260915-r1.log`.

| Artifact | SHA-256 |
| --- | --- |
| `preflight.json` | `ef5e7c7c5a3b1ab9941456e33956f9fa8b36dad55eea8e1dad49639dba6ffdc3` |
| `result.json` | `72093da04ff15b01f16dee8fa565de7b2483af421f74ea5d07ac9529271f5dd8` |

The executed command used `--enable-provider`; replay omits it:

```powershell
.\.pixi\envs\dev\python.exe -X utf8 -m tools.audit_native_spine_question_quality --root eval_results/native-spine-question-quality-20260915-r1
```

## Next action

The goal remains active and below 95%. Continue on this same cached history.
The isolated reader change is a measured improvement; it leaves ten misses with
incomplete recorded support and nine with all recorded support. Inspect a generic
summary-routing change that prevents weak lexical prefixes from seeding unrelated
parent chunks, and a larger bounded context that retains later user statements.
Use existing attention-derived topology and exact raw addresses; do not send raw
text to Qwen, add question-specific source rules, remove failing questions, or
resume any 100-history controller.

Any evaluation-item repair must be versioned and reported separately from system
improvement. A subsequent threshold pass is insufficient for goal completion
without checking actual source correctness and the documented ambiguities.
