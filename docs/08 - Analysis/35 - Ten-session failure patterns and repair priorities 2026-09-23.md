# Ten-session failure patterns and repair priorities

**Status:** CURRENT — measured diagnosis; proposed repairs remain unimplemented.  
**Date:** 2026-09-23.  
**Applies to:** The completed ten-history, 1,000-question user-spine memory evaluation.  
**Depends on:** [Research Log 244](../10%20-%20Research%20Log/244%20-%202026-09-22%20-%20Ten%20million-token%20session%20evaluation.md) and the [sealed evaluation artifacts](../../eval_results/native-spine-ten100-20260922-r1/aggregate-report.json).

**913/1,000 answers passed the unchanged grader.** The 87 marked misses reveal
specific weaknesses in retaining decisive user statements, interpreting updates,
and selecting among similar episodes. They also include verified grading errors.
The most useful next work is to repair these boundaries while retaining the
measured compact context and seconds-level response time.

## Measurement boundary

Research Log 244 records the frozen configuration, application ingest/reopen
lifecycle, and complete accuracy, latency, and token measurements. This analysis
uses that completed run's saved source turns, answer packets, and judgments;
it made no new model calls and does not isolate attention's contribution.

All 87 marked question/prediction/reference comparisons were reviewed.
Representative cases below were also checked against original source turns,
served context, or saved judge explanations. The four patterns overlap; they
are not an exhaustive, disjoint classification of all 87 misses. No adjusted
accuracy is reported.

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
