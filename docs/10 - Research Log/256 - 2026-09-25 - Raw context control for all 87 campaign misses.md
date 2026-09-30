# Raw context control for all 87 campaign misses

**Status:** Complete — every miss from the ten-session battery checked.  
**Date:** 2026-09-25.  
**Applies to:** All 87 historical misses from the 913/1,000 campaign across ten million-token histories.  
**Depends on:** [Campaign failure analysis](../08%20-%20Analysis/35%20-%20Ten-session%20failure%20patterns%20and%20repair%20priorities%202026-09-23.md), [Log 255](255%20-%202026-09-25%20-%20Full%20hundred%20question%20packet%20versus%20raw%20context%20control.md).

**Raw context receives incorrect labels on 21 of the original 87 misses:
24.1%.** It passes 57; nine remain unresolved. Excluding those nine, the raw
failure fraction is 21/78, or 26.9%. These are source-review labels, not proof
that 21 questions are inherently unanswerable by the model.

The original memory answers were also reviewed against the same full sources:
40 are labeled correct, 38 incorrect and nine unresolved. Among the 38 still
labeled incorrect, **13 also fail with raw context (34.2%)**, while **25 pass
with raw context**. The reviewer itself has questionable judgments, including
a clear name-normalization false positive described below; “still labeled
incorrect” must not be read as independently confirmed ground truth.

The earlier 75% result applied to four fresh misses on one repaired cap-8
history. This experiment uses the **original battery's 87 failed answers and
original packets**, before that repair. It answers the broader question without
changing the historical miss denominator or regenerating the memory answers.

## Complete results

| Source-review label | Original memory answers | Direct raw answers |
| --- | ---: | ---: |
| Correct | 40 | 57 |
| Incorrect | 38 | 21 |
| Ambiguous | 7 | 7 |
| Invalid paired review | 2 | 2 |
| Total original misses checked | 87 | 87 |

| Paired outcome | Count |
| --- | ---: |
| Both correct under source review | 32 |
| Both incorrect | 13 |
| Memory incorrect, raw correct | 25 |
| Memory correct, raw incorrect | 8 |
| Both ambiguous | 7 |
| Invalid paired review | 2 |

The 21 raw failures include eight cases where source review accepts the original
memory answer. Consequently, **21/87** measures raw failure among historically
flagged questions, whereas **13/38** measures shared failure among memory
answers the new reviewer still rejects. They answer different questions.
Neither fraction changes the original 913/1,000 benchmark: the 913 passing
answers have not been re-reviewed here, and the grading method differs.

Every history is represented; question membership was checked exactly against
the sealed campaign aggregate:

| History | Original misses | Raw correct | Raw incorrect | Unresolved |
| --- | ---: | ---: | ---: | ---: |
| 01 | 6 | 4 | 1 | 1 |
| 02 | 15 | 8 | 4 | 3 |
| 03 | 12 | 10 | 1 | 1 |
| 04 | 5 | 2 | 1 | 2 |
| 05 | 7 | 6 | 1 | 0 |
| 06 | 7 | 5 | 2 | 0 |
| 07 | 12 | 6 | 4 | 2 |
| 08 | 7 | 7 | 0 | 0 |
| 09 | 11 | 7 | 4 | 0 |
| 10 | 5 | 2 | 3 | 0 |
| Total | 87 | 57 | 21 | 9 |

Ambiguous cases are H1 Q50; H2 Q12, Q42 and Q75; H3 Q12; H4 Q50; and H7 Q37.
H4 Q75 fails exact-source-quote validation; H7 Q82 fails prediction/reference
anchor validation. Invalid reviews are preserved, not retried or counted as
passes. All 87 raw answers and all 87 paired review responses exist.

## Evidence delivery versus reader behavior

The original campaign's recorded-support classification was verified again.
Sixty-one missed packets contained all recorded support quotes; twenty-six
did not. This is coverage of the recorded support, not a guarantee that the
reference enumerates every fact relevant to the question.

| Original packet support | Cases | Memory incorrect now | Raw incorrect now | Both incorrect | Raw recovers memory | Raw-only incorrect | Unresolved |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| All recorded quotes present | 61 | 18 | 18 | 11 | 7 | 7 | 4 |
| Some recorded quotes missing | 26 | 20 | 3 | 2 | 18 | 1 | 5 |

**Eighteen of the 25 raw recoveries come from incomplete-support packets.**
Among the twenty still-rejected memory answers in that group, eighteen pass
with raw context. This is strong diagnostic evidence that evidence delivery
accounted for many failures in the original battery.

When the original packet already contained all recorded quotes, raw context
does not improve the aggregate incorrect count: eighteen in each arm, with
seven recoveries and seven regressions. Eleven of the eighteen rejected memory
answers also fail with raw context, or **61.1%**. That group is more consistent
with reader interpretation, completeness and grading problems. These results
do not establish how the current repaired cap-8 policy performs across all ten
histories; its full campaign has not been rerun.

### What persists, and what the labels miss

The thirteen shared failure labels are H2 Q21/Q54, H3 Q30, H6 Q23/Q51,
H7 Q63/Q67, H9 Q38/Q78/Q80, and H10 Q6/Q14/Q83. Inspection distinguishes
useful findings from overly literal or unstable judgments:

- **Plan selection, H6 Q23:** Both answers select the earlier idea of attending
  city council meetings instead of the later stated plan to join a local
  sustainability group. The raw answer had both statements available.
- **Coverage, H9 Q38:** Raw identifies the purchased James Parker print that
  memory could not identify, but omits other art experiences requested by the
  broad question. A binary shared-failure label hides meaningful improvement.
- **Technical coverage, H6 Q51:** Memory says “I don't know.” Raw identifies the
  note-count discrepancy, sharp-note support and octave-parsing error, but is
  rejected for omitting the question about a simplified calculation's
  equivalence. This is not evidence that raw context was equally unhelpful.
- **Intent/scope disputes:** H2 Q21 penalizes removing “it looks like” from the
  graduation statement; H2 Q54 penalizes including the proposed linen lampshade;
  H3 Q30 disputes whether the Space Needle plan includes Emily. These are
  weaker examples of a substantive failure than an omitted fact or wrong plan.
- **Question/reference problems:** H9 Q78 interprets “last Saturday” relative
  to the synthetic question date, although the source purchases were described
  earlier. H9 Q80 penalizes omitting that a practice exercise was requested
  again. Neither should be casually presented as a hard model capability limit.
- **Clear grading defect, H10 Q14:** The memory answer's only flagged issue is
  changing “John Hopkins University” to “Johns Hopkins University.” I would not
  treat that name normalization as a meaningful recall error. The raw answer
  additionally names assistant-supplied architectures as user interests, so its
  label has a separate attribution issue. Recorded model-review labels are
  retained unchanged; no post-hoc manual score is substituted.

The eight raw-only failure labels are H1 Q74, H2 Q11/Q28, H4 Q23, H5 Q47,
H7 Q71/Q98, and H9 Q92. Several repeat the earlier attribution pattern: raw
answers import assistant suggestions into user requirements, such as an
under-$15 brunch target or extra necklace specifications. H9 Q92 omits the
user's current peanut-butter ingredient even though it is in the full source
and the original memory answer includes it. H7 Q98 instead penalizes adding
an airport name/code to “the airport,” another debatable scope distinction.

Reviewer variability also matters. The six reused history-01 raw answers are
byte-identical to those in Log 255, yet this new paired review accepts Q66
and calls Q50 ambiguous where the earlier review rejected them. Their memory
counterparts differ because this experiment uses the original battery's
answers. Exact-quote validation establishes quotation fidelity, not consistent
semantic judgments. The measured fractions describe this review pass rather
than an irreducible model-error rate.

## Matched inputs and scope

Both answer arms use the same `codex_sdk/gpt-5.6-sol` route, the original v7
reader, the same dated question, and a requested 256-token answer cap.
The memory answer is the exact saved campaign response. The raw answer replaces
only its evidence block with complete original source conversations, including
all user and assistant turns, timestamps and speaker labels. The comparison
therefore uses historical memory answers and newly generated raw answers, not
a contemporaneous fresh rerun of both arms.

Raw selection includes every conversation represented in the served memory
spans and adds the reference-origin conversation if absent. Six required that
addition: H3 Q56/Q60, H4 Q50, H7 Q51, and H8 Q2/Q91. This gives the raw model
the missing source in those cases; reference answers themselves are never
shown to the answerer. It is an evidence-availability diagnostic, not an
independent test of retrieving from all 1M tokens.

This uses **complete selected conversations, not entire million-token prompts**.
There are 365 conversation instances across the 87 contexts, with repetition
between questions. All **1,095 served memory spans** were authenticated against
the original raw turns, and a separate audit verifies **all 213 recorded support
quotes appear in the raw inputs**. No context is truncated.

The same Terra source-review model judges each A/B pair against identical full
sources, exact original memory excerpts and the fallible reference. Arm order
alternates; identities are hidden, although the source contents can provide
clues. All raw answers were sealed before reviews began. Reviewer output cap is
3,072 tokens. Three raw calls and then three reviews may be in flight. The
reader, grading instructions and input construction were frozen before calls.

## Tokens, artifacts and verification

| Per-answer tokens, provider reported | Original memory | Raw context |
| --- | ---: | ---: |
| Mean input | 1,637.80 | 11,795.36 |
| Maximum input | 2,657 | 26,763 |
| Mean output | 41.49 | 50.80 |

Memory uses **86.11% fewer answer-input tokens** on this failed-question subset.
This excludes historical ingestion and experimental review costs. There is
no new production latency claim.

The run reused 87 original memory answers and six raw answers whose model,
messages and output cap matched exactly. It made **168 new provider calls**:
81 raw answers and 87 paired reviews, using **2,124,933 input and 36,606 output
tokens**. All newly generated responses ended with `stop`. No retries, new
ingestion, retrieval, Qwen inference, DSPy generation or production edits occurred.

Canonical artifacts:
[`eval_results/native-spine-ten100-misses-raw-control-20260925-r1`](../../eval_results/native-spine-ten100-misses-raw-control-20260925-r1).
Per-case files use `hNN-qNNN` with one-based question numbers. The report includes
all 87 cases, history totals, support-coverage groups and exact response bindings.

- Preflight: `5c13bc4e958f985a39bc1142ad600648d1e3f2707387b495aed97577a8c843d0`.
- Complete raw-answer seal: `68483b6c06c410cff51d0741707ac1d82df467a76990ad407e34c57370371d7d`.
- Support audit: `188bb2084d0e778d4925004beeced700cdad9e5e2c68e30aecf1e49f77a13f8e`.
- Report: `cbcf04e191cce9fc63ac2c0ebd735fa8c109965fe113ef77a79347c13f62e203`.

Provider-disabled replay reproduced the identical report hash and authenticated
the complete original miss set, source-bank reconstruction, reader/question
preservation, original predictions, exact cached requests, reviewer evidence,
all call hashes and the 168 reservations. The preparation launch used `runpy`,
so replay retains that form to preserve its recorded script-path identity.

From the repository root with `PYTHONPATH=src;.` and the usual UTF-8 and
offline-model environment settings:

```powershell
eval_results/native-spine-dspy-hints-20260924-r1/venv/Scripts/python.exe -u -c "import runpy; runpy.run_path('eval_results/native-spine-ten100-misses-raw-control-20260925-r1/run.py',run_name='__main__')" report
.pixi/envs/dev/python.exe eval_results/native-spine-ten100-misses-raw-control-20260925-r1/support_audit.py
```

These commands must reproduce the hashes without new model calls. Use the
coverage split to distinguish evidence-delivery work from reader work, and
resolve grading defects before claiming a new whole-system accuracy score.
