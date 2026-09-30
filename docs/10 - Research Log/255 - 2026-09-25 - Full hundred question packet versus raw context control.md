# Full hundred question packet versus raw context control

**Status:** Complete — 75% of fresh memory misses also fail with raw context.  
**Date:** 2026-09-25.  
**Applies to:** All 100 questions on the existing 1,115,343-token history.  
**Depends on:** [Log 254](254%20-%202026-09-25%20-%20Original%20conversation%20control%20for%20reader%20errors.md), [Log 253](253%20-%202026-09-25%20-%20Expanded%20DSPy%20hint%20evaluation%20on%20100%20saved%20questions.md).

The user asked to extend the original-context control across the 100-question
run and measure which memory misses also fail when the frontier answer model
receives raw context directly. **Three of four fresh memory misses also fail
with raw context: 75%.** Following the eight plain-memory misses from the
previous 100-question run instead yields **four of eight: 50%**. These are
different denominators and must remain separate.

Both arms use the same `codex_sdk/gpt-5.6-sol` answer model and v7 reader.
Memory supplies its saved plain cap-8 packet; the direct arm supplies complete
original source conversations, including all user and assistant turns. This
compares context delivery to the same LLM. It is not Qwen versus Sol, and
there are no DSPy hints in either arm.

## Full paired result

The twenty completed pairs from Log 254 are retained unchanged. The remaining
eighty receive new paired answers and reviews under the same protocol. No
question is dropped or selected based on its new answer.

| Source-review label | Memory packet | Original conversations |
| --- | ---: | ---: |
| Correct | 95 | 87 |
| Incorrect | 4 | 12 |
| Ambiguous | 1 | 1 |
| Invalid review | 0 | 0 |

The paired outcomes are:

| Outcome | Questions | IDs where relevant |
| --- | ---: | --- |
| Both correct | 86 | |
| Both incorrect | 3 | Q6, Q65, Q66 |
| Memory incorrect, original correct | 1 | Q56 |
| Memory correct, original incorrect | 9 | Q34, Q50, Q60, Q63, Q64, Q67, Q74, Q82, Q96 |
| Both ambiguous | 1 | Q46 |

The predeclared metric is `both incorrect / memory incorrect = 3 / 4 = 75%`.
There are no ambiguous raw verdicts among those four memory misses. The other
25% is one question that passes with raw context. On assessable questions,
labels are 95/99 versus 87/99. These are model source-review judgments over an
already examined development history, not a replacement for the older binary
benchmark or independent proof of the 95% accuracy target.

### Following the earlier misses

| Earlier population | Earlier misses | Raw incorrect now | Raw correct now | Raw unresolved now |
| --- | ---: | ---: | ---: | ---: |
| Log 253 plain-memory source review | 8 | 4 (50%) | 4 | 0 |
| Log 253 incorrect in either plain or DSPy arm | 12 | 6 (50%) | 6 | 0 |
| Log 248 binary benchmark | 6 | 2 (33.3%) | 3 | 1 |

The most recent plain-memory misses were Q6, Q12, Q42, Q54, Q63, Q65, Q66 and
Q70. Raw context still receives incorrect labels on Q6, Q63, Q65 and Q66.
The other four now pass **in both arms**. Fresh memory also passes Q63, so five
of the eight historical memory misses no longer reproduce in memory itself.
Consequently, the four historical passes cannot be credited to raw context.
Q63's new raw failure concerns different wording from its earlier memory
failure; the 50% follow-up is question-level recurrence, not necessarily
recurrence of the identical mistake.

The older 94/100 binary benchmark uses a different grader and has different
misses: Q23, Q35, Q46, Q50, Q66 and Q68. This source review flags raw answers
Q50 and Q66 and leaves Q46 ambiguous. Its denominator must not be substituted
for the latest eight-miss population or the four fresh paired misses.

## What the misses mean

The three shared flags are unchanged from Log 254:

- **Q6:** Both expand the quoted fee-payment enumeration. The reviewer's
  distinction is debatable and is not treated as a confirmed legal error.
- **Q65:** Both repeat a three-month yoga duration recorded on March 4 when
  answering an April 5 question. The reference repeats that duration too.
  A more careful answer would date the recorded duration and qualify any
  assumption about continued attendance.
- **Q66:** Both omit Strava while naming the bike computer and heart-rate
  monitor. Strava is present in both inputs, but whether the question's
  "tracking equipment" wording requires an app remains a scope dispute.

**Q56 is the sole memory-only flag.** The memory answer includes attending
upcoming Egypt exhibitions at the Met. The source asks about exhibitions and
says the user would like to learn more, while explicitly expressing interest
in attending lectures or events. The raw answer sticks to the latter.
Both statements were verified present in the memory packet. The reviewer
penalizes an intent inference, not missing evidence; whether the inference is
materially wrong is also debatable.

The nine original-context-only flags have several patterns:

| Cases | Observed distinction |
| --- | --- |
| Q60, Q64, Q67, Q82 | Raw answers add an unstated yoga progression, intended lentil dishes, clothing-brand requirements, or unarchiving capability. Q64 and Q67 directly illustrate assistant suggestions becoming user intentions or requirements. |
| Q50, Q96 | Raw answers omit a later weekday-breakfast update or a concern about harmful dependence on technology that memory includes. Q50 also exposes an incomplete reference answer. |
| Q34 | Raw combines a gardening plan from a separate invasive-species discussion with the requested mouse-conservation response. |
| Q63, Q74 | Raw weakens the boss-conversation plan or omits consideration of discounts. These small scope/wording judgments deserve less weight; Log 254 records reviewer inconsistencies. |

The practical result is that original context does not automatically remove
reader errors. Some raw-only failures reflect attribution or coverage problems
even when all source text is supplied. However, **75% is a fraction of four
grader-labeled misses**, not an estimate of permanently unsolvable questions
or proof that model weights alone are responsible. The reader prompt is held
fixed, generation is one pass per arm, and some labels depend on question or
reference interpretation. No production change follows from these labels alone.

## Control boundaries and tokens

This is a **complete selected-source-conversation control**, not a direct
1M-token prompt. Source selection follows the served packet's provenance,
with the same predeclared provision to add a missing reference-origin
conversation. No reference conversation needed adding in any of the 100
questions. Reference answers were never passed to either answerer.

The audit reconstructed **352 conversation instances** across the question
contexts and authenticated all **1,625 served memory spans** as exact subsets
of their original raw turns. Conversation counts include reuse across questions.
Every original user and assistant turn is retained, without truncation. The
control changes context length, layout, ordering and assistant content together;
it does not independently ablate those factors or test retrieval from all 1M
tokens without source selection.

| Per-answer tokens, provider reported | Memory | Original context |
| --- | ---: | ---: |
| Mean input | 1,761.60 | 9,288.28 |
| Maximum input | 2,526 | 24,349 |
| Mean output | 30.93 | 32.04 |

Memory uses **81.03% fewer answer-input tokens** than these original-context
prompts. This excludes historical ingestion and experimental review costs.
The test does not establish a new production latency measurement.

## Execution and verification

The extension made exactly **240 new calls**: 160 Sol answers and 80 Terra
paired source reviews. Combined with the retained twenty pairs, the complete
comparison contains 200 Sol answers and 100 Terra reviews. All 300 responses
ended with `stop`. The extension used **1,680,344 input and 19,496 output
tokens**; both stages together used 2,177,260 input and 28,880 output tokens.
All 100 reviews pass exact-quote validation. No requests were retried.

All new answers were sealed before new grading. New answer execution used two
pairs in flight, and reviews used three. Arm order alternated. Both answers
were graded against identical complete raw sources, authenticated memory
excerpts and the fallible reference, with A/B arm identities hidden. The
review evidence can still reveal clues about which arm produced an answer.
Reader, question, output cap and model were fixed. No history was rebuilt;
there were no new retrieval, Qwen or DSPy calls and no production edits.

The initial report check stopped after all calls completed because it compared
serialized review JSON to data reloaded from canonical sorted-key storage.
Field order differed although all 100 payloads were structurally identical.
The frozen `run.py`, preflight and all requests remain unchanged. The separate
provider-disabled `audit.py` compares parsed JSON while still validating the
exact original message hashes. It also checks original prompt substitution,
source reconstruction, reference bindings, predictions, review validation,
the full call set and reservation counts. This repairs reporting only.
A second provider-disabled replay reproduced the identical report hash with
the reservation count still at 240. `git diff --check` also passed.

Canonical artifacts:
[`eval_results/native-spine-raw-context-full100-20260925-r1`](../../eval_results/native-spine-raw-context-full100-20260925-r1).
The report binds all one hundred pairs, including the twenty retained from
Log 254, and reports both the fresh and historical miss denominators.

- Preflight: `97a77e46ce6e5ccda525d54c5c868be5b620ccc1d21614395843e544f10e623f`.
- Complete answer seal: `a269f28f37804a1e244550c33fbee134c664d01fc36c6be80f08d5ad94609e24`.
- Audit method: `f036e7127ba42dd1c988b1254e1115e391e3ffc63e66a9b30531da8e265ee41e`.
- Report: `34911c58cd1eaea0b9cf3a1373cc4227b21d592aaabe79ea94900260b9bba611`.

Run from the repository root with `PYTHONPATH=src;.` and the usual UTF-8 and
offline-model environment settings:

```powershell
eval_results/native-spine-dspy-hints-20260924-r1/venv/Scripts/python.exe eval_results/native-spine-raw-context-full100-20260925-r1/audit.py
```

The provider-disabled replay must reproduce the report hash without changing
the 240-call reservation count. Use the paired counts to assess context-delivery
differences; retain ambiguous and disputed examples when considering any reader
change instead of treating every model-review label as settled ground truth.
