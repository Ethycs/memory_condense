# Lightweight hints on reader failures and passing controls

**Date:** 2026-09-25  
**Status:** Completed diagnostic; positive targeted result, not enabled in production  
**Depends on:** [Log 256](256%20-%202026-09-25%20-%20Raw%20context%20control%20for%20all%2087%20campaign%20misses.md), [Log 253](253%20-%202026-09-25%20-%20Expanded%20DSPy%20hint%20evaluation%20on%20100%20saved%20questions.md)

## Result

Lightweight hints improved the 12 selected reader-error candidates from **2/12
correct to 5/12**, with **three recoveries and no correct-to-incorrect changes**.
All **18 previously passing controls remained correct in both fresh arms**.
Hints added **244.64 input tokens per answer, or 16.07%**, with **zero hint-generation
model calls**. This supports using compact summary navigation to address some
reader errors, but does not establish a gain across the full application workload.

| Frozen group | Questions | Plain correct / incorrect / ambiguous | Hinted correct / incorrect / ambiguous |
| --- | ---: | ---: | ---: |
| Reader-error candidates | 12 | 2 / 10 / 0 | 5 / 7 / 0 |
| Historical passing controls | 18 | 18 / 0 / 0 | 18 / 0 / 0 |
| Material evidence gaps | 4 | 0 / 3 / 1 | 1 / 2 / 1 |
| Grading diagnostics | 2 | 1 / 1 / 0 | 1 / 1 / 0 |
| All selected cases | 36 | 21 / 14 / 1 | 25 / 10 / 1 |

All 36 paired reviews passed exact-source-quote and prediction-anchor validation.
These are source-review labels from a selected diagnostic population, not a
replacement for the original **913/1,000** benchmark. Every answer is fresh;
historical grades determine selection only.

The three primary recoveries are:

- **H2 Q5:** retains the requested **100** Kepler quiz questions, alongside the
  non-multiple-choice question/hint/answer format.
- **H6 Q23:** selects the later stated plan to **join a local sustainability
  group**, instead of the earlier tentative city-council idea. This is the
  clearest change in which fact the reader selects.
- **H7 Q63:** includes **English**, alongside the requested subtle, indirect
  feedback wording.

The seven shared failure labels are H2 Q21, H3 Q15/Q30, H4 Q59, H7 Q67,
and H10 Q71/Q83. Hints do not consistently fix omitted follow-ups or selecting
the wrong conversation. For example, the Holborn rooftop requirement and agile
industry-adaptation request are explicitly present in both the raw packet and
the new navigation labels, yet both answers still omit them.

## Population and grading boundaries

Log 256 identified 18 memory answers labeled incorrect despite containing all
recorded reference-support quotes. Checking the actual evidence cited for each
failure revealed that this coverage flag is weaker than complete answer evidence.
Before generating any answers, the 18 were divided into:

- **12 reader candidates:** the quoted content underlying the prior complaint
  is present in the served packet.
- **Four evidence-gap cases:** H2 Q30 lacks the tea/herbal-blend request; H3 Q74
  lacks later kitchenware/beauty shopping plans; H8 Q83 lacks the selected Wild
  Magic origin; H10 Q6 lacks the completed Stranger Things season. Some also
  contain a separate reader error on evidence that is present.
- **Two grading diagnostics:** H9 Q78 has a question/source relative-date
  conflict; H10 Q14 was previously rejected solely for normalizing “John” to
  “Johns Hopkins.” Neither contributes to the primary recovery score.

The 18 controls are original binary passes with complete recorded support:
one per history, then eight additional cases, selected by a fixed salted hash.
They cover all ten existing million-token histories. No histories are rebuilt.

Both fresh arms are reviewed against the same complete original conversations
and exact memory excerpts, with A/B labels and alternating arm order. References
are fallible aids and are never given to the answerer or hint selector. Before
calls, the rubric was clarified equally for both arms: harmless name spelling
normalization is not a contradiction, details must follow the actual question,
and conflicting relative dates require attention to question ambiguity. This
is not a direct comparison of fresh labels with Log 256's old labels.

Review limitations remain visible:

- H2 Q30 receives a hinted pass for restoring the yoga-pose/video request,
  although both answers still omit the unserved tea request. The new reviewer
  does not penalize that omission. This fourth graded recovery remains outside
  the primary reader result, as frozen before calls.
- Both NVC exercise answers, H9 Q80, pass. Adding that the exercise was repeated
  receives no extra recovery credit.
- Both H10 Q14 answers pass after the spelling clarification.
- H10 Q71 remains debatable: the source says both “I got them” and “I'm still
  deciding.” The reviewer rejects both answers for omitting the acquisition
  statement. Those recorded labels are retained.
- Both H9 Q78 answers fail on the date conflict; the hinted answer also adds an
  unsupported claim that cereal and milk were Walmart purchases. Zero binary
  regressions does not mean every changed claim improved.

## What the hints do

The isolated runner builds a partial directory from the **current question and
stored summaries of served turns**, with authenticated turn IDs, speaker roles
and recording dates. It chooses up to six turns by distinct query-word overlap
with summaries, uses stable source-order ties, and copies up to 16 words from
each chosen summary. Rows appear in original packet order. The guide is capped
at 320 tokens; the observed maximum is 261.

A short fixed instruction distinguishes recording dates from event dates,
requests from completed actions, and assistant suggestions from user intent.
It directs the reader to verify claims in the excerpts and consider unlisted
follow-ups. The directory is partial navigation, not an exhaustive checklist.

The hint selector does not reason over raw text, references, or reviewer findings.
Raw text is accessed only to authenticate served spans and preserve the existing
answer prompt. Removing the guide reconstructs the original prompt exactly:
same **Sol route, v7 reader, dated question, raw excerpts and 256-token output cap**.
The answerer still receives the raw memory excerpts in both arms.

There is **no DSPy optimization or model-generated hint call** in this variant.
The earlier DSPy test motivates avoiding that extra call; the populations differ,
so this run is not a direct accuracy comparison against DSPy. It tests a combined
directory/instruction change, not the separate contribution of each component.

These are the original campaign's saved packets, **before the cap-8 repair**.
No retrieval, ingestion, Qwen inference or production reader policy changes occur.
The result supports a further check on current cap-8 packets before promotion;
it does not establish a new full-system score or solve missing evidence.

## Tokens and verification

| Provider-reported mean per answer | Plain | Hinted |
| --- | ---: | ---: |
| Input tokens | 1,522.28 | 1,766.92 |
| Output tokens | 34.39 | 34.53 |

The extra 244.64 input tokens require no additional network round trip. This
concurrent diagnostic does not measure interactive latency. All **108 provider
calls** completed: 72 Sol answers and 36 Terra paired reviews, using **565,823
input and 13,656 output tokens**, including evaluation overhead. No retries
occurred. Actual response routes match Sol/Terra, and every response ended with
`stop`.

All answers were sealed before reviews began. The completed report authenticates
**418 served raw spans**, reconstructs original source conversations, checks
unchanged evidence and question framing, regenerates deterministic guides,
validates response/request hashes, and accounts for all 108 reservations.
Provider-disabled replay reproduced the identical report hash.

Canonical artifacts:
[`eval_results/native-spine-lightweight-hints-20260925-r1`](../../eval_results/native-spine-lightweight-hints-20260925-r1).

- Preflight: `a27094c50cd82860c4949f5635847d9938e17e4af2f5bd9e48ecb09b05d44a94`.
- Complete answer seal: `56c6dce79ce5dbfd5003d327a67d6eedcece985543d7822cb06eb2e6ec3ef284`.
- Report: `813a0215df46a414832701527d0926710db6bf1fbc7193c4d901b56a4d865bd0`.

From the repository root, replay without provider access:

```powershell
$env:PYTHONPATH='src;.'
$env:PYTHONUTF8='1'
$env:KMP_DUPLICATE_LIB_OK='TRUE'
$env:HF_HUB_OFFLINE='1'
$env:TRANSFORMERS_OFFLINE='1'
eval_results/native-spine-dspy-hints-20260924-r1/venv/Scripts/python.exe -u eval_results/native-spine-lightweight-hints-20260925-r1/run.py report
```
