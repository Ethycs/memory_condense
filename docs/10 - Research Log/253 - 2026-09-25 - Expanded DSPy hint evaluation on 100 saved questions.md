# Expanded DSPy hint evaluation on 100 saved questions

**Status:** Complete — no consistent improvement on the newly added population; not promoted.  
**Date:** 2026-09-25.  
**Applies to:** All 100 saved cap-8/v7 packets from the existing 1,115,343-token history.  
**Depends on:** [Log 252](252%20-%202026-09-24%20-%20DSPy%20optimization%20of%20saved-packet%20hints.md).

The user requested a larger test of DSPy hints against reader errors in the
returned evidence. This experiment expands from twelve paired questions to
the complete existing 100-question population. Every question gets fresh
plain and hinted answers; no history is rebuilt and no retrieval is rerun.

**The larger test does not establish a consistent benefit.** On the 82 questions
new to DSPy evaluation, the two arms tie: **74/78 assessable answers correct**,
with two recoveries and two regressions. Across all 94 questions outside the
training set, the labels favor hints **86/90 versus 83/90**. That net gain comes
from three of the twelve questions already included in the earlier soft test.
The frozen prompt improves some precision and coverage examples while causing
others to lose details or broaden a requirement incorrectly.

## Results

| Population | Questions | Assessable pairs | Plain correct | DSPy correct | Recoveries / regressions |
| --- | ---: | ---: | ---: | ---: | ---: |
| Primary: outside DSPy training | 94 | 90 | 83/90 (92.2%) | 86/90 (95.6%) | 5 / 2 |
| Newly added to DSPy evaluation | 82 | 78 | 74/78 (94.9%) | 74/78 (94.9%) | 2 / 2 |
| Training diagnostics | 6 | 6 | 5/6 | 3/6 | 0 / 2 |
| All questions, including training | 100 | 96 | 88/96 | 89/96 | 5 / 4 |

**Four pairs are unresolved in both arms:** Q46 is judged ambiguous, and
Q22, Q83 and Q85 fail exact-source-quote validation. They are included in the
population counts, excluded from the assessable denominators, and never treated
as passes. The raw primary-population labels are therefore 83 correct, 7
incorrect and 4 unresolved for plain; 86 correct, 4 incorrect and 4 unresolved
for DSPy. These are source-review results, not replacements for the previous
binary-grader benchmark score or proof that the system meets its 95% target.

The primary paired comparison has five recoveries versus two regressions
(exact paired binomial p = 0.4531). The newly added population has two versus
two. One pass on one reused history, with several debatable judgments, is
insufficient to establish a reliable accuracy improvement.

### What changed in the errors

The source-review issue counts on the 94 questions outside training are:

| Issue kind | Plain | DSPy |
| --- | ---: | ---: |
| Unsupported claim | 4 | 1 |
| Contradiction | 1 | 0 |
| Missing requested detail | 1 | 1 |
| Scope mismatch | 1 | 2 |
| Ambiguous scope | 1 | 1 |

These are issue instances, not an independent accuracy score. Hints show a
signal on precision, but do not consistently resolve completeness or scope.
The concrete pairs explain the tradeoff:

| Case | Population | Source-review change | Inspection |
| --- | --- | --- | --- |
| Q6: quoted fee-payment requirement | New | Recovered | The reviewer prefers the hinted answer's narrower list of accompanying forms. Whether the plain wording is materially different is debatable; this is not a strong recovery example. |
| Q12: game-starting plan | Previously evaluated, outside training | Recovered | Hinted preserves a tentative stealth-first plan; plain turns “maybe” into a definite desire. |
| Q54: purchase timing | Previously evaluated, outside training | Recovered | Hinted attaches “last week” only to the book purchase; plain also attaches it to the sneaker purchase. The original sentence's modifier scope permits interpretive disagreement. |
| Q63: burnout self-care | New | Recovered | Hinted includes the stated plan to try a new hobby, possibly painting. Plain omits it despite the turn being present in the raw packet. |
| Q70: living-room changes | Previously evaluated, outside training | Recovered | Hinted avoids converting a request for planter recommendations into a decided use of complementary planters. |
| Q23: art-exhibition themes | New | Regressed | Hinted omits abstract art, which plain includes. Its guide points to environmental/social themes and community artwork, overlooking another requested theme in the raw text. |
| Q67: sustainable clothing | New | Regressed | Hinted imports “long-lasting” from a tentative luxury-goods discussion as a definite clothing-brand requirement. Its guide explicitly highlights that neighboring preference. |
| Q62: historical-fiction topics | Training | Regressed | Hinted adds “Britain's” to SOE. The reviewer rejects the unstated detail under the closed-source evaluation rules; this is not a demonstrated false historical fact. |
| Q74: botanical-garden plans | Training | Regressed | Hinted turns a possible gardening-tool purchase plus a separate gift-shop-discount question into buying the tools from the gift shop. |

On the nine questions flagged by the earlier model-comparison source review,
fresh plain answers receive 9/9 correct labels versus 7/9 for hints. The two
losses are training cases Q62 and Q74. Most historical failures did not recur
in the fresh plain arm, so this subset cannot demonstrate a recovery rate for
the old mistakes. Model output and reviewer variability remain relevant.

Both arms still omit Strava in the cycling diagnostic Q66. The reviewer rejects
both for that omission in this run, although earlier reviews disagreed about
whether the question's “equipment” scope requires the app. This is another
reason to retain examples and source quotes alongside aggregate labels.

The practical pattern is that a guide can direct the reader to an overlooked
turn, as in Q63, but it can also make its selected topics look exhaustive or
encourage combining neighboring situations, as in Q23 and Q67. The raw evidence
remains intact in every case. This is an observed reader behavior with hints,
not evidence of a new retrieval loss.

### Token and runtime cost

All **100 guides passed validation**, with no fallback packets. The declared
status alias was used on 37 questions.

| Per-question mean | Plain | DSPy |
| --- | ---: | ---: |
| Answer-reader input | 1,761.60 tokens | 1,892.00 tokens |
| Answer output | 31.06 tokens | 31.44 tokens |
| Additional hint-generation input | 0 | 1,587.34 tokens |
| Additional hint-generation output | 0 | 65.86 tokens |
| Combined generation + reader input | 1,761.60 tokens | 3,479.34 tokens |

The guide adds **130.4 reader-input tokens (+7.40%)**. Generating it brings
combined input to approximately twice the plain memory packet. The additional
hint call averaged **6.94 seconds under four-pair concurrency**; this is not a
production latency benchmark, and no end-to-end latency improvement is claimed.
This query-time implementation has not earned its additional cost. The existing
memory-serving path remains in place.

## Frozen comparison

The selected DSPy COPRO instruction and output prefix from Log 252 are frozen.
There is no additional optimization or few-shot training. Hint and answer
generation use the same Sol gateway route, and the answer reader remains v7
with a requested 256-token output cap. The hint generator sees only the current
question, persisted summaries, turn IDs, roles and recording timestamps.
It never receives raw evidence or evaluation references.

One mechanical format correction is declared before evaluation: `future_goal`
is accepted as an alias for `goal` and rendered as `goal`. This resolves the
previous optimizer/validator mismatch without changing the learned prompt or
inferring new content. All other guide limits remain in place: at most six
entries, short purpose labels, known turn IDs and a 320-token local-proxy cap.
Any invalid guide is recorded and replaced by the unchanged original packet;
there is no regeneration attempt. Results therefore evaluate the frozen prompt
with this declared renderer correction, not the exact Log 252 validator.

The original system prompt, question, raw evidence and excerpt order are
preserved in both answer arms. Guides add metadata outside the raw-evidence
budget. Timestamps are copied from authenticated metadata. Preparation also
checks that all recorded reference-support quotes were present in the previously
audited cap-8 packets. This is an experiment on how the reader uses delivered
evidence, not a retrieval or ingestion comparison.

## Populations and grading

- **Primary population: 94 questions excluded from DSPy training.** Twelve
  were in the previous soft test; this run produces fresh answers for them.
- **New expansion: 82 of those questions** were in neither the six-question
  training set nor the twelve-question DSPy soft test.
- **Training diagnostics: Q1, Q30, Q62, Q66, Q74 and Q94.** These six are
  reported separately and are not evidence of generalization.
- **Historical issue subset:** Questions flagged by the earlier source-grounded
  model comparison, using labels fixed before this run. These are selected
  diagnostic cases, not an independent population score.

All cases come from the same previously examined history. The 82-question
expansion is new to DSPy evaluation, not newly collected data. Aggregate
100-question results are descriptive because they include the training cases.

Plain and hinted execution order alternates. Four question pairs may be in
flight. All 200 answers must be sealed before the grader opens references.
Terra reviews each pair against the same authenticated raw sources and fallible
reference. Reviewer A/B order alternates; the grader receives neither the guide
nor arm identities. Exact source-quote validation is required. An invalid
review remains unresolved and is not silently counted correct or retried.

The report records paired recoveries and regressions, answer-issue kinds,
ambiguous questions, invalid reviews, rejected guides and token overhead.
Source-review labels are model judgments, not independent human adjudications.
Differences were also inspected against the saved sources to assess whether
they concern substantive coverage or merely small wording choices.

## Execution artifacts

Canonical root:
[`eval_results/native-spine-dspy-expanded100-20260925-r1`](../../eval_results/native-spine-dspy-expanded100-20260925-r1).
The runner reuses the isolated DSPy 3.4.0 environment from Log 252 and writes
new sealed artifacts without changing the earlier experiment.

The run completed exactly **400 provider requests**: 100 hint generations, 200
answers and 100 paired source reviews. Automatic network retries were disabled.
Hint generation uses a requested 768-token output cap; reviews use 2,048.
No GPU model is loaded, and there are no new Qwen calls. Production code and
dependencies are unchanged by this experiment.

All 400 responses ended with `stop`. Provider-reported totals, including
evaluation, were **767,032 input and 33,583 output tokens**. The report audit
verified every request/response binding, all 100 summary-only hint inputs and
preservation of the original question, reader and raw evidence for all pairs.
Provider-disabled report replay reproduced the exact report hash with zero new
calls. No new model training, ingestion or 1M-history rebuild occurred.

Preflight SHA-256:
`ab4cdbae6778262cf7863fd53da6725fd0b66d8d743240b6445149e2eecd59c6`.

- Complete answer seal: `cbd7c25c7f0cd28bf5ca6047e254008c96af465d6274047649294589607d8739`.
- Report: `32f4ddbda4b896e514527502b726be3551e63d53074001b38b1f05328eef7d52`.

Run from the repository root, with `PYTHONPATH=src;.` and the usual UTF-8 and
offline-model environment settings:

```powershell
eval_results/native-spine-dspy-hints-20260924-r1/venv/Scripts/python.exe eval_results/native-spine-dspy-expanded100-20260925-r1/run.py answer --enable-provider
eval_results/native-spine-dspy-hints-20260924-r1/venv/Scripts/python.exe eval_results/native-spine-dspy-expanded100-20260925-r1/run.py grade --enable-provider
# Offline validation and exact report replay after completion:
eval_results/native-spine-dspy-hints-20260924-r1/venv/Scripts/python.exe eval_results/native-spine-dspy-expanded100-20260925-r1/run.py report
```
