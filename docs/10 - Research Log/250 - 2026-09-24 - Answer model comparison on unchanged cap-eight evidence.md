# Answer model comparison on unchanged cap-eight evidence

**Status:** Complete — 100 candidate answers, 100 original grades, 84 paired source reviews; no model change promoted.  
**Date:** 2026-09-24.  
**Applies to:** The same 100-question, 1,115,343-token history and cap-8 packets as Log 248.  
**Depends on:** [Log 248](248%20-%202026-09-23%20-%20Cap-eight%20repair%20on%20one%20complete%20million-token%20history.md) and [Log 249](249%20-%202026-09-23%20-%20Cap-eight%20integration%20and%20reader%20scope%20comparison.md).

The alternative answer route scores **91/100**, against **94/100** for the saved
Sol answers, with identical evidence and v7 instructions. Both models omit
Strava from the same bike-tracking answer. This comparison does not establish
that replacing the answer model solves the remaining problems. Retain Sol and
v7; the integrated cap-8 routing remains unchanged.

## Model availability and execution

The current local gateway inventory was captured before model selection. It
lists the direct `claude-opus-5` route and `claude_code/claude-opus-4-7`, alongside
the existing Codex SDK routes. The first direct Opus request failed with HTTP
400 because its Anthropic account had insufficient credits. No answer was
generated, and the other 99 requests were never submitted. That failed attempt,
its frozen plan and its evaluator source are preserved separately under
`eval_results/native-spine-answer-model-cap8-20260924-r1`.

The completed candidate uses **`claude_code/claude-opus-4-7`**. Its first answer
completed before the remaining bounded batch was released. All 100 responses
report that requested alias. These are authenticated gateway route identities;
the upstream model checkpoints and SDK wrappers were not independently inspected.

`tools/evaluate_native_spine_answer_model_cap8.py` reuses the saved-packet protocol,
streaming measurement function, binary grader and bounded dispatcher. It requires
every complete system/evidence/question prompt to match the original cap-8
request. It preserves the v7 reader, requested 256-token output cap, questions,
references and Sol grader. The first request is serial; the rest run at concurrency
four. There are 100 fresh completed answers and 100 fresh grading calls in this
successful run, with no retries. No ingestion, retrieval or Qwen pass is rerun.

This is a comparison against historical Sol answers, not a simultaneous control
or a new latency benchmark. The comparison discipline also follows the general
[OpenAI evaluation guidance](https://developers.openai.com/api/docs/guides/evals)
to assess changed model outputs against fixed application expectations.

## Fixed-grader results

| Measurement | Saved Sol + v7 | Claude Code Opus + v7 |
| --- | ---: | ---: |
| Graded correct answers | 94/100 | 91/100 |
| Mean provider-reported input tokens | 1,761.60 | 1,761.60 |
| Mean provider-reported output tokens | 31.17 | 46.15 |
| Answers ending with `stop` | 100 | 100 |
| Maximum reported output tokens | 126 | 319 |

Sixteen answer strings are identical; 84 differ. The candidate gains **Q35** and
loses **Q19, Q56, Q63 and Q96**. No byte-identical answer receives opposite
grades. Remaining failed questions are Q19, Q23, Q46, Q50, Q56, Q63, Q66, Q68
and Q96. Every original grade remains intact.

**Output-limit caveat:** Both requests specify 256 output tokens, but the Claude
Code route returns **319 tokens on Q94**, ending with `stop`. The local output
token count confirms the reported count. This route did not enforce the limit
on that answer. Q94 passes in both runs and is not one of the score changes.
Do not describe the experiment as having an identical enforced output budget,
or infer a pure underlying-model effect from a comparison of gateway routes.

## Source inspection of the changed outcomes

- **Q35, physics summary:** Both answers supply the requested 223-word bulleted
  format. Opus additionally repeats the reference's concepts-and-equations
  requirement. Its passing grade does not demonstrate that Sol failed the
  explicitly asked format/count question.
- **Q19, stand-up writing:** Opus covers the stated difficulties and notebook
  practice but omits testing jokes on friends and family, which Sol included.
  That practice is present in the evidence. The question specifically asks for
  constraints and challenges, so whether testing is mandatory is a scope issue.
- **Q56, Met activities:** Both answers include exhibitions/events and
  lectures/workshops/courses. Only Opus is penalized for exhibitions/events.
  The served user text explicitly asks about upcoming exhibitions or events
  related to ancient Egypt and expresses interest in learning more. This is
  inconsistent treatment of supported content.
- **Q63, burnout:** Opus includes starting a hobby, possibly painting. The user
  explicitly proposes painting in the same self-care conversation and then says
  they are excited to try it. The grade rejects it because the reference omits
  it; it is not an invented plan.
- **Q96, technology concerns:** Opus omits excessive dependence and possible
  harm, while Sol includes them. The evidence explicitly asks whether technology
  could become detrimental to society and whether people could become too
  dependent on it. This is a substantive completeness regression.
- **Q66, shared miss:** Both models name the Garmin Edge 130 and separate
  heart-rate monitor but omit the user's existing Strava use. The exact user
  statement is in the first selected conversation. Another answer model did
  not recover this detail. The question's use of “equipment” may also encourage
  a narrower reading than the reference's inclusion of an app.

## Paired source review

The source review covers all 84 changed answer pairs. The other 16 answers are
identical and pass both original grades. `tools/review_native_spine_model_pairs.py`
uses Terra to inspect both saved answers together against their exact served
context and the reference conversation's original user turns. The latter are
authenticated against the raw bank and clearly labeled as potentially unserved.
No new answer is generated.

The reviewer receives neither model names nor original grades. A fixed salted
hash assigns each model to A or B per question. The existing source-review
schema and exact-quote validator apply separately to both answers, rejecting
invented quotes, incorrect answer anchors and malformed output. An invalid pair
stays invalid without retry. Findings are diagnostic and do not replace either
100-question score.

All **84 review calls** completed. Provider-free replay uses 84 cache hits and
reproduces the same report hash. The review labels are:

| Diagnostic label on the 84 selected pairs | Sol | Opus |
| --- | ---: | ---: |
| Correct | 78 | 77 |
| Incorrect | 4 | 5 |
| Ambiguous | 1 | 1 |
| Invalid pair | 1 | 1 |

These are not replacement accuracy rates. Q12 has a nonexact purported source
quote, invalidating the pair. Q53 has multiple translation requests with
different target languages and an underspecified question. Neither is excluded
from the 84-pair denominator.

The reviewer accepts all six original Sol failures and flags different issues
among answers the original grader passed. It flags Sol-only issues on Q1, Q30
and Q62; Opus-only issues on Q90, Q94, Q96 and Q99; and a shared omission of
plant-species exploration on Q74. Some judgments depend on how narrowly one
reads qualifications or question scope. This is evidence that the original
reference-matching score does not isolate reader quality, not proof that every
new diagnostic label is right.

Direct source inspection supports the Q96 omission. It also supports the Q94
scope problem: Opus turns a remembered lazy Sunday into a routine being
established, despite the user's subsequent goal of consistent early wake-ups.
Q94 is the same answer that exceeded the requested output limit. More output
did not guarantee better fidelity.

**Correction to the earlier Strava diagnosis:** The diagnostic reviewer accepts
both Q66 answers. Both supply the decided physical tracking equipment, while
the question says “equipment” and the reference additionally requires an app.
Strava's omission is an observable difference from the reference, but describing
it as an unambiguous reader failure was too strong. It does not establish that
either base model is unable to extract the stated fact.

The secondary reviewer is also fallible: its Q19 rationale says the Opus answer
includes testing material, although that answer omits the friends/family testing
step. Exact quote validation checks provenance, not the correctness of every
inference in an assessment. Its labels remain diagnostic, with that limitation
recorded rather than silently repaired.

## Verification and artifacts

Twenty-three focused model-comparison and source-review tests pass. The initial
comparison/reader checks also passed seven tests. Prompts are preserved in full,
references open for grading only after all answers seal, and no source or
production reader policy is modified by either comparison.

- Successful run: `eval_results/native-spine-answer-model-cap8-20260924-r2`.
- Inventory SHA-256: `2922a5babc838b2dff8edf43915e0be00454a24871687febf856b1af4a3cdcd7`.
- Preflight SHA-256: `3139427f738988d4a32441cf58528fd40f820cd42ebce2bff5c5d39bc1d1f4cc`.
- Fixed-grader report SHA-256: `a4ddb2250afa1d527212204736afd9bfe274fdc7e37e743328067796aa030714`.
- Transport audit SHA-256: `5f70410f336b0bcba27262bf120ca1adb210a5d07d6e4464289ba96e1dc8535d`.
- Source-review preflight SHA-256: `3cf064647c7e11fd141814627f6ddbc7bb782ced1ad6455a9bbeb960ea16c331`.
- Source-review report SHA-256: `b01950c58c3b59243a4f6a199913249607e40ecd974be370982ccff430685d7d`.
- Both score and source-review reports replay byte-identically with zero new provider calls.

Provider-free replay:

```powershell
$env:PYTHONPATH = 'src;.'
$env:PYTHONUTF8 = '1'
& .pixi/envs/dev/python.exe -m tools.evaluate_native_spine_answer_model_cap8 report
& .pixi/envs/dev/python.exe -m tools.review_native_spine_model_pairs
```

This one exposed history does not establish generalization or a base-model
capacity ceiling. The observed errors include answer omissions, interpretation
of question scope, and defects in reference-based grading. The original
ten-session score remains 913/1,000.
