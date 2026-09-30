# Original conversation control for reader errors

**Status:** Complete — original conversations do not remove the shared failure labels.  
**Date:** 2026-09-25.  
**Applies to:** Twenty diagnostic questions from the existing 1,115,343-token history.  
**Depends on:** [Log 253](253%20-%202026-09-25%20-%20Expanded%20DSPy%20hint%20evaluation%20on%20100%20saved%20questions.md).

The user requested a control using actual context to determine whether an LLM
makes the same mistakes without memory compression. We generated fresh answers
with the same Sol model and v7 reader, comparing saved memory packets with the
complete original conversations from which those packets were drawn.

**The reviewer flags three questions as incorrect in both arms. No question
fails with memory and passes with original context.** Overall labels are
16 correct, 3 incorrect and 1 ambiguous for memory, versus 13 correct,
6 incorrect and 1 ambiguous for original context. These are diagnostic
source-review labels; several distinctions are debatable. They do not establish
a general accuracy advantage for memory.

## Results

| Selected population | Questions | Memory correct | Original context correct | Both incorrect | Ambiguous in both |
| --- | ---: | ---: | ---: | ---: | ---: |
| Incorrect in either previous DSPy arm | 12 | 9 | 6 | 3 | 0 |
| Previously unresolved | 4 | 3 | 3 | 0 | 1 |
| Passing controls | 4 | 4 | 4 | 0 | 0 |
| All selected cases | 20 | 16 | 13 | 3 | 1 |

On the 19 assessable pairs, the labels are 16/19 versus 13/19. All four passing
controls remain correct. Q46 remains ambiguous because the nutrition question
does not specify which of several meal-planning exchanges it covers. All twenty
reviews pass exact-quote validation; none is excluded for an invalid review.

### What persists with original context

| Case | Shared answer behavior | Interpretation of the review |
| --- | --- | --- |
| Q6: fee-payment requirement | Both list I-129 alongside the accompanying forms as requiring separate payments. | The reviewer objects to expanding the source's enumeration. Whether this is materially wrong is debatable; it is not a strong factual-error example. |
| Q65: yoga duration | Both answer approximately three months, copying the March 4 statement for an April 5 question. | Both overlook the time distinction. The reference also says three months and is flagged by the reviewer. A qualified answer would distinguish the recorded duration from any assumption of continued attendance. |
| Q66: cycling setup | Both name the Garmin Edge 130 and separate heart-rate monitor, omitting Strava. | Strava is explicitly present in both inputs. Whether an app is required by a question asking for tracking "equipment" remains a scope dispute. |

These behaviors recur despite complete original conversations. They support a
reader-interpretation or evaluation-scope explanation for these cases, rather
than loss of the relevant raw statements. The experiment holds the reader
prompt fixed and therefore does not separate model behavior from prompt effects.

### Original-context-only failure labels

Q67 provides the clearest substantive example. The user requests high-quality,
eco-friendly, affordable and stylish clothing. The original-context answer adds
transparency and fair labor practices as the user's requirements. Inspection
finds those terms in **assistant recommendations** in the supplied transcript;
the memory answer sticks to the user's stated requirements. More raw dialogue
creates an opportunity to misattribute assistant suggestions even with speaker
labels and the same attribution-aware reader prompt.

The other two differences deserve less weight:

- **Q63:** Original context says the user would "consider talking" to their
  boss; memory says they would talk to their boss. The reviewer prefers the
  latter because a later user turn states a plan. Both answers omit the hobby
  example that the previous review penalized. The differing coverage judgments
  show reviewer variability; this is not a stable measure of recovered detail.
- **Q74:** Original context omits the question about promotions or discounts;
  memory includes it. Whether that detail is essential to the requested
  activities and purchases is debatable. The original answer also says tools
  would be purchased from the gift shop, a connection the prior review
  criticized but this reviewer did not flag.

The three-label advantage must not be presented as three independently
confirmed engineering improvements. The useful finding is that providing
the original conversations does not reliably cure these answer behaviors,
and can introduce attribution mistakes of its own.

## Matched control

The population was fixed before new calls: all twelve previously incorrect
cases (Q6, Q12, Q23, Q42, Q54, Q62, Q63, Q65, Q66, Q67, Q70, Q74), all four
unresolved cases (Q22, Q46, Q83, Q85), and four deterministically selected
passing controls (Q1, Q16, Q45, Q94). This is a selected diagnostic sample,
not an independent estimate of overall system accuracy.

For each question, the control includes every complete source conversation
represented in the served memory spans, with every user and assistant turn
verbatim, in chronological conversation order, with timestamps and speaker
labels. The predeclared selection rule also allowed adding the reference-origin
conversation if absent. **Zero reference conversations needed adding.**
Reference answers were never passed to either answerer.

Across the twenty packets, 78 conversation instances contain 438 user turns
and 441 assistant turns. These counts include repeated conversations across
questions, not newly ingested histories. All **356 served memory spans** were
verified against their original source IDs, turn identities, roles, dates,
full-turn hashes and exact character slices. No original context was truncated.

This is a **complete source-conversation control**, not a direct run over the
entire 1M-token history. Source selection still uses the existing memory
provenance. The comparison asks whether removing excerpt selection within those
conversations remedies reader errors; it does not test retrieval independently
or isolate the effects of length, assistant content, layout and ordering.

Both arms use `codex_sdk/gpt-5.6-sol`, the unchanged v7 system prompt, the same
dated question and answer instruction, and a requested 256-token output cap.
Only the evidence block is replaced. Memory receives its original plain cap-8
packet, without DSPy hints. Execution order alternates, with two pairs in flight.

After all forty answers were sealed, `codex_sdk/gpt-5.6-terra` reviewed each
A/B pair against identical complete sources, the authenticated memory excerpts,
and the fallible reference. Arm names were hidden and A/B order alternated;
the shared evidence can still give clues about the arms. Exact source quotes
are required, but successful quote validation does not establish that each
semantic judgment is sound. Reviews use a requested 3,072-token output cap,
with three pairs in flight. No retries or post-result prompt changes were made.

Earlier grades were used only to select cases. The fresh answers and fuller
review sources mean the old DSPy scores cannot be directly compared with these
scores as an improvement over time.

## Tokens and verification

| Mean per answer, provider reported | Memory | Original context |
| --- | ---: | ---: |
| Input tokens | 1,865.10 | 10,669.40 |
| Output tokens | 51.25 | 54.15 |
| Maximum input tokens | 2,346 | 24,349 |

Memory uses **82.52% fewer answer-input tokens** on this selected sample. This
excludes prior ingestion costs and the experimental reviewer. No new production
latency claim is made from these concurrent diagnostic calls.

Exactly **60 provider calls** completed: forty Sol answers and twenty Terra
reviews. All finished with `stop`; total experimental usage was **496,916 input
and 9,384 output tokens**. There were sixty reservations and no failed or retried
requests. No ingestion, retrieval, Qwen inference, DSPy optimization or production
code changes occurred.

Provider-disabled report replay reconstructed every original conversation,
reverified all served spans and call bindings, and reproduced the identical
report SHA-256 with no new calls.

Canonical artifacts:
[`eval_results/native-spine-raw-context-control-20260925-r1`](../../eval_results/native-spine-raw-context-control-20260925-r1).
This contains the frozen runner, preflight, exact paired prompts, answers,
source reviews, sixty recorded calls and final report.

- Preflight: `0037494eaf95f98bbb031081cdd0a648fbde1857d3db08c5a47b6b11c3f88641`.
- Complete answer seal: `658729cf421d33cae2af77479d0dfeee78e0b1c6aff4650e374112eafd84668e`.
- Report: `0eb9063f068f6d24e423460b1797ebdac707440808e38e0b8ff6d05217812123`.

Offline replay from the repository root, with `PYTHONPATH=src;.` and the
usual UTF-8 and offline-model environment settings:

```powershell
eval_results/native-spine-dspy-hints-20260924-r1/venv/Scripts/python.exe eval_results/native-spine-raw-context-control-20260925-r1/run.py report
```
