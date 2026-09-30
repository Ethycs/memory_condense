# Ten-session failure patterns and repair priorities

**Status:** CURRENT — measured diagnosis. Priority 1 traced, repaired in code and checked on the 87 misses plus matched controls (27 recovered, 2 controls lost); the full-campaign effect is unmeasured. See [Research Log 245](../10%20-%20Research%20Log/245%20-%202026-09-23%20-%20Earliest%20loss%20trace%20and%20user%20completion%20routing.md).  
**Date:** 2026-09-25 (complete classification added; original analysis dated 2026-09-23).

**Applies to:** The completed ten-history, 1,000-question user-spine memory evaluation.  
**Depends on:** [Research Log 244](../10%20-%20Research%20Log/244%20-%202026-09-22%20-%20Ten%20million-token%20session%20evaluation.md) and the [sealed evaluation artifacts](../../eval_results/native-spine-ten100-20260922-r1/aggregate-report.json).

**913/1,000 answers passed the unchanged grader.** The 87 marked misses reveal
specific weaknesses in retaining decisive user statements, interpreting updates,
and selecting among similar episodes. They also include verified grading errors.
The most useful next work is to repair these boundaries while retaining the
measured compact context and seconds-level response time.

## Complete classification of the 87 original misses — September 25

All 87 now have a disjoint, repair-oriented classification based on the saved
original packets and the completed source review in
[Log 256](../10%20-%20Research%20Log/256%20-%202026-09-25%20-%20Raw%20context%20control%20for%20all%2087%20campaign%20misses.md).
This is an analyst diagnosis of that review, not another model run or a change
to the original benchmark grades.

| Primary class | Count | Share of 87 | Evidence |
| --- | ---: | ---: | --- |
| Original answer accepted on source review | 40 | 46.0% | The later reviewer accepts the saved memory answer against the original sources; these are not confirmed memory failures. |
| Evidence delivery gap | 24 | 27.6% | Relevant content is absent from the packet, or the required source conversation was not selected. Some also have reader issues. |
| Reader-error candidate with cited evidence present | 12 | 13.8% | The source text underlying the complaint is in the packet, but the answer omits, misattributes, or misinterprets it. |
| Identified grading/question defect | 2 | 2.3% | H9 Q78 has a relative-date question/reference conflict; H10 Q14 is rejected for John/Johns Hopkins spelling normalization. |
| Ambiguous question | 7 | 8.0% | Multiple source-supported scopes prevent a dependable binary grade. |
| Invalid source-review response | 2 | 2.3% | Required quotation or prediction/reference anchoring fails validation. |
| **Total** | **87** | **100%** | Every original campaign miss appears exactly once. |

Source-review acceptance is not independent human adjudication, and several
reader-candidate judgments remain debatable. This table does not establish a
revised whole-system accuracy: the 913 original passes were not audited with
this review method.

### Evidence delivery: 24 cases

**Nineteen** have a relevant source conversation represented in the packet but
omit necessary content within the selected conversations. **Five** omit the
reference-origin conversation entirely: H3 Q56/Q60, H7 Q51, H8 Q2/Q91. The latter
include cabinet-quiz instructions, the farm-budget request, a different book
recommendation episode, exercise plans, and Pollinations formatting requirements.
This identifies where evidence is absent; it does not independently locate every
loss in summarization, ranking, routing, or hydration.

Other examples include the final Garmin/heart-rate-monitor decision (H1 Q66),
the standalone Wild Magic choice (H8 Q83), technical MIDI requirements (H6 Q51),
and an itinerary's nearby-place clustering instruction (H9 Q72).

**Twenty-one of these 24 pass with complete raw source conversations.** The
remaining three—H6 Q51, H9 Q38, H10 Q6—receive additional correct details with
raw context but still fail on incomplete coverage. A delivery problem can coexist
with a reader problem; delivery takes precedence in this disjoint repair table.

The number 24 differs from the earlier quote-coverage counts for a specific
reason. Of 26 historical misses with missing recorded quotes, 20 remain rejected
under source review; one is accepted and five are unresolved. Four more rejected
cases have missing required facts outside their recorded support quotes:
H2 Q30, H3 Q74, H8 Q83, and H10 Q6. Thus **20 + 4 = 24**. The old support checklist
did not cover every fact needed by the question.

Consequently, the earlier **18/25 raw recoveries from incomplete recorded
support** becomes **21/25 with an identified material delivery gap** after
examining the actual complaints. The remaining four recoveries are among the
12 reader candidates. These are two classifications of the same 25 recoveries,
not additional recovered questions.

### Reader interpretation: 12 candidates

| Class | Count | Cases and observed issue |
| --- | ---: | --- |
| Omitted available detail | 6 | H2 Q5: 100 quiz items; H3 Q15: rooftop/city view; H4 Q59: adapting agile by industry; H7 Q63: English feedback; H9 Q80: repeating the NVC exercise; H10 Q83: long-term effects of Trajan's campaigns. |
| State, intent or certainty | 3 | H2 Q54: proposed lampshade treated as selected; H6 Q23: earlier tentative council idea chosen over later group-joining plan; H10 Q71: definite non-purchase despite conflicting purchase/consideration statements. |
| Episode or participant scope | 2 | H3 Q30: individual Space Needle plan attributed to the trip with Emily; H7 Q67: wrong lunch-request episode selected although the intended request is present. |
| Temporal interpretation | 1 | H2 Q21: email receipt date treated as the date graduation clearance occurred. |

Four pass with raw context; eight still fail. These are **candidates**, not 12
indisputable capability failures. The NVC repetition requirement, the lamp's
“leaning toward” wording, the travel participant scope, and inconsistent boots
statements expose grading or interpretation uncertainty as well.

The later fresh comparison in
[Log 257](../10%20-%20Research%20Log/257%20-%202026-09-25%20-%20Lightweight%20hints%20on%20reader%20failures%20and%20passing%20controls.md)
uses these same 12 candidates. Plain answers pass 2/12 and lightweight hints
pass 5/12, with three recoveries and no binary regressions; all 18 historical
passing controls pass in both fresh arms. That measures hints on unchanged
packets, separately from using summary cues to retrieve missing material.

### Remaining review outcomes and reproducibility

The seven ambiguous cases are H1 Q50; H2 Q12/Q42/Q75; H3 Q12; H4 Q50; H7 Q37.
The invalid pairs are H4 Q75 and H7 Q82. All 40 source-review-accepted cases and
every other case are individually listed in the
[87-row CSV](../../eval_results/native-spine-ten100-misses-raw-control-20260925-r1/failure-classification.csv).
The companion
[JSON](../../eval_results/native-spine-ten100-misses-raw-control-20260925-r1/failure-classification.json)
retains the original prediction, review bindings, complaints, and checks of
whether quoted complaint evidence occurs in the served packet.

The provider-free classifier verifies sealed artifact hashes, exact membership
in the original 87-miss set, all category totals, and the evidence-presence checks.
It removes only local C/T source-numbering tags when checking quoted text;
it does not rewrite transcript content or any historical grade.
Replay from the repository root:

```powershell
.pixi/envs/dev/python.exe -X utf8 eval_results/native-spine-ten100-misses-raw-control-20260925-r1/failure_classification.py
```

Classification JSON SHA-256:
`be4dde16e700d519cfbc4982af3bee210cbc4c3d3f05a678559e133bd26685b6`.
CSV SHA-256:
`a7bba42aa179c88573238ff58e9965f1c98e44025bd52d05a0901b4fd45fb3eb`.
No new model calls, ingestion, retrieval, or production changes were needed.

## Measurement boundary

Research Log 244 records the frozen configuration, application ingest/reopen
lifecycle, and complete accuracy, latency, and token measurements. This analysis
uses that completed run's saved source turns, answer packets, and judgments;
it made no new model calls and does not isolate attention's contribution.

In the original September 23 analysis below, all 87 marked
question/prediction/reference comparisons were reviewed.
Representative cases below were also checked against original source turns,
served context, or saved judge explanations. The four patterns overlap; they
are not themselves an exhaustive, disjoint classification. The September 25
section above supplies that classification; the original observations below
remain useful examples. No adjusted accuracy is reported.

## Quote coverage is useful but incomplete

| Recorded support quotes in answer context | Correct | Incorrect | Total |
| --- | ---: | ---: | ---: |
| All present | 911 | 61 | 972 |
| One or more absent | 2 | 26 | 28 |
| Total | 913 | 87 | 1,000 |

Every recorded support quote is present for **97.2% of questions**, and for
**70.1% of marked misses**. Missing quotations strongly coincide with failure
in this sample: 26 of the 28 such answers were marked incorrect. However, neither
row measures complete answer evidence. Authors recorded at most three support
quotes, and a reference can contain additional facts outside those quotes.

All 1,000 packets passed exact raw reconstruction. That verifies fidelity of
the selected text. A faithful packet can still omit a required turn, retrieve
the wrong episode, or contain conflicting states that the reader mishandles.
Consequently, the 61 misses with all recorded quotes cannot simply be assigned
to the answer model, and 97.2% cannot be reported as complete factual recall.

## Pattern 1: decisive fragments and final updates disappear

| Case | Original evidence | Served context and resulting answer |
| --- | --- | --- |
| H1 Q66, road-bike equipment | Final turn says Garmin Edge 130 is on the way and selects a separate compatible heart-rate monitor | Earlier Garmin consideration is present; final turn is absent; answer says undecided |
| H8 Q83, D&D character | Standalone user turn chooses `wild magic` | Race, class, and ability scores survive; subclass choice does not; answer omits Wild Magic |
| H6 Q51, text-to-MIDI work | Sharp-note code, octave parsing of `#`, and 38 versus 44 notes | Mostly general music conversations plus one theme-request turn; answer abstains |
| H3 Q56, cabinet quiz | User corrects the format to short answers with initials as hints | Format requirements absent; answer abstains |
| H8 Q91, Pollinations formatting | User requires natural-English translation and omission of the label `output` | Required source instructions absent; answer abstains |

The D&D case is particularly diagnostic: all three recorded support quotes
arrived, but none covered Wild Magic. The final answer could not recover the
missing choice from that packet. The Garmin case shows the same structural
problem with a longer update: selecting the topic without its conclusion
changes the apparent decision state.

**Observed boundary:** necessary source content failed to reach the reader.
**Still unresolved:** whether each loss occurred during summarization, ranking,
parent expansion, or final projection. The source-to-packet comparison alone
does not establish which upstream component caused it.

## Pattern 2: decisions are misread even when both states arrive

In **H6 Q23**, the packet contains an initial idea to attend city-council
meetings and a later statement, in the same conversation, about starting by
joining a local sustainability group. The reader chooses the earlier idea.
Here, more retrieval is not required to expose the later decision: it is
already in the served context.

In **H2 Q54**, the user chooses a wooden lamp finish but asks whether a linen
shade would work. The answer treats the shade as part of the chosen lamp.
This loses the distinction between a commitment and an option still being
considered.

## Pattern 3: repeated topics and question scope compete

Marked misses repeatedly concern breakfasts, books, exercise, shopping, and
community events. Similar content from another conversation can replace a
required fact or expand the answer with details outside the intended episode.
Some references were authored from one source episode while their questions
sound like requests about the whole history.

Two source-checked examples show why these cases need careful interpretation:

- **H7 Q37, baseball count:** The reference expects 15 baseballs from July. The
  packet also contains a December statement about adding 20 baseballs, and the
  actual question is dated the following January without identifying July.
  The answer says 20. This is not an established correct answer: additions and
  total holdings differ. It exposes both ambiguous episode scope and a reader
  distinction that needs preserving.
- **H6 Q33, comedy class:** On May 29, the user says the class began six weeks
  earlier. The question is dated June 7. `About 7 weeks` is consistent with
  advancing elapsed time to the question date, whereas the reference repeats
  `6 weeks`. The judge rejects the numeric difference without source dates.

Added answer details require source and relevance checks before being
classified as hallucinations.

## Pattern 4: correct concise answers fail an overbroad reference

The saved judge prompt supplies the question, gold answer, and prediction.
It does not supply the original source. Although its instructions allow extra
detail, they also require the same facts and reject missing facts. In the
following five verified cases, the judge demands a reference detail outside
the narrower question actually asked.

| Case | Question asks | Candidate answer | Omission cited by judge |
| --- | --- | --- | --- |
| H1 Q35 | Summary format and word count | 223 words, bulleted list | Concepts and equations covered |
| H2 Q28 | Necklace type and style | Citrine, solitaire pendant | Around $40 |
| H3 Q45 | Which game first | Azul | Codenames later |
| H5 Q65 | What was finished | Clay figurine of cat Luna | It turned out well |
| H5 Q97 | Engagement rate | Around 10% | Higher than the usual rate |

These are demonstrable grading false negatives, not proposed system repairs.
Other marked misses may have similar problems, but five examples do not
establish the full count. A review of failed answers alone also does not check
whether any accepted answers were wrongly graded correct.

The original **91.3%** remains the recorded score; this review does not
establish 95% accuracy.

## Repair priorities and bounded validation

1. **Trace and preserve missing decisions.** Start with H8 Q83, H1 Q66, and
   H6 Q51 using existing source, summary, routing, hydration, and rendered
   artifacts. Identify the earliest stage that loses the needed information.
   Preserve short choices with their surrounding exchange, relevant later
   updates, and distinctive technical terms in routing summaries. A candidate
   fix should preserve exact source spans, respect the context budget, and
   avoid displacing evidence on previously successful controls.
2. **Improve state and episode interpretation.** Use H6 Q23 and H2 Q54 as
   evidence-present diagnostics, with ambiguous cases kept separate. Preserve
   considered/chosen/completed distinctions and answer each requested facet.
   Bind facts to the entity, episode, and time the question specifies. A later
   explicit decision can supersede a proposal; a later question is not a
   decision. Preserve ambiguity when the question does not identify an episode.
   Compare reader behavior on fixed packets before widening retrieval.
3. **Separate evaluation defects from engineering errors.** Review question
   scope, source support, required facts, and temporal framing. Distinguish
   required facts from optional context, using source evidence to adjudicate
   disputed additions and dates. Keep the sealed verdicts and report revised
   methodology and results separately; include accepted answers in any broader
   accuracy audit. Fixing grading does not repair missing evidence.
4. **Confirm the stable candidate at the existing operating point.** Develop
   against a small set of saved cases plus passing controls before a larger
   run. Reuse persisted histories and cached compilation where appropriate;
   rebuilding histories is needed only when the change affects ingestion or
   stored representations. Record accuracy, input length, and median/p95
   response time together. An inspected development set cannot demonstrate
   held-out generalization.

These are proposed engineering priorities, not completed changes. A larger
answer model could help interpretation, but it cannot recover a user choice
absent from its supplied context. The observed examples therefore support
focused evidence and state-handling work before a broad model or architecture
replacement. They provide no causal attribution to individual attention heads.

## Evidence locations and verification

All case labels above use **one-based question numbers**. Under
`eval_results/native-spine-ten100-20260922-r1/history-NN/`, saved answer filenames
use **zero-based ordinals**: H8 Q83 is `history-08/answers/082.response.json`.
Its original turns and supports are entry 82 of `questions/references.json`.
`judge-preflight.json` binds question ordinals to judge messages; responses in
`judge-checkpoints/` are matched by their `messages_sha256` field.

The [aggregate report](../../eval_results/native-spine-ten100-20260922-r1/aggregate-report.json)
and [readable misses](../../eval_results/native-spine-ten100-20260922-r1/report.md)
retain the original measurements and answers. This document is the canonical
failure analysis. From the worktree root, this
read-only PowerShell command verifies the sealed aggregate and reproduces the
coverage table without model calls:

```powershell
@'
import hashlib
import json
from collections import Counter
from pathlib import Path

root = Path('eval_results/native-spine-ten100-20260922-r1')
aggregate = root / 'aggregate-report.json'
assert hashlib.sha256(aggregate.read_bytes()).hexdigest() == (
    'bdd4542fc53deff6506b319fb8757d39afeaa42694222c2df20f91b22e30ced5'
)
reports = [json.loads((root / f'history-{h:02d}' / 'report.json')
                     .read_text(encoding='utf-8')) for h in range(1, 11)]
assert all(len(report['rows']) == 100 for report in reports)
rows = [row for report in reports for row in report['rows']]
counts = Counter((row['correct'], row['all_recorded_quotes_in_context'])
                 for row in rows)
assert counts == {(True, True): 911, (False, True): 61,
                  (True, False): 2, (False, False): 26}
print('Correct: 913/1000; misses: 87; miss quotes present/absent: 61/26')
print(dict(counts))
'@ | .\.pixi\envs\dev\python.exe -X utf8 -
```

**Decision after verification:** trace a demonstrated missing turn to its
earliest loss, then choose an evidence-selection repair or a reader repair
based on whether the required fact reached the actual answer prompt.
