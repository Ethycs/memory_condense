# Fine user summary retrieval and compact full100 comparison

The preceding relative-day repair improved its fresh full100 control from
76 to 78 correct answers. This successor addresses a broader selection issue:
the coarse router backfills other user turns from selected conversations,
including turns that are weak matches for the actual question.

## Candidate

`search/fine_spine_routing.py` scores stored user-fragment summaries directly.
The maximum fragment score ranks each whole user turn. The query uses the same
BGE encoder as the existing summary indexes. Raw transcript text, source IDs,
question IDs and reference labels are not embedding inputs. All dates remain
subject to the existing inclusive question-day cutoff.

Date-reserved user turns stay first. Eight direct user matches precede two
attached-context sections from the existing plan. The remaining direct matches,
up to 32 user turns total, precede the old attached-context tail. Exact duplicate
sections are removed. Hydration still validates the original whole user turns
and selected attached sections; it enforces both token and span budgets.

The new index contains 24,566 user fragments representing 24,518 whole user
turns across the same ten complete approximately 1.04M-token memories. It
reuses their already generated summaries. No transcript resummarization,
new Qwen calls or raw embedding inputs are introduced. Qwen's existing
attention-partitioned hierarchy remains bound to the source coordinates.

Sixteen focused tests pass, including complete user-fragment coverage,
summary-only compiler inputs, exact whole-turn hydration, future exclusion,
date reservation, all three full100 populations and the answer-before-gold
barrier.

## Frozen inputs and comparison

Index root: `eval_results/full1m-fine-spine-addresses-20260910-r1`.

- Compilation preflight:
  `fcd9be5723cb5b2407553b17af3dded111faecdcf24442132b4434958bf4cdd9`.
- Completed ten-index population:
  `ac6e483b9a5062f944d8fd9bce92bb5acc860b1083e31d829553f428ab65df42`.

Evaluation root: `eval_results/full1m-fine-spine-packets-full100-20260910-r1`.

The three arms are the current relative-reservation control, fine retrieval
with 3,072 context tokens, and fine retrieval with 2,048 context tokens. All
use the same v2 reader, Terra model, 256-token output cap, temperature zero,
concurrency eight and no retries. Temperature zero applies to this entire
fresh comparison; the previous Log 169 reader requests omitted temperature.
Every original control packet is reproduced live during preparation. Identical
reader prompts may share a fresh response; previous predictions are not reused.
All 300 logical answers must seal before Sol judging.

This is an accuracy screen. Smaller packet size is not proof of API-like
latency. The joint >=95/100 accuracy and latency gate remains required on the
same fresh streamed predictions before the overall goal can pass. The separate
native corpus remains parked, and the confirmation population is untouched.

Packet preparation completed for all100, reproducing every control exactly.
The evaluation preflight SHA is
`79a5cdaef5db7edc1d2106cd648324a80611a8c8f4503fcf7c5e68824ec0e883`.
All 300 reader prompts are distinct and require fresh responses. Median
candidate context sizes are 3,043.5 and 2,030.5 tokens respectively; all stay
within their declared budgets.

## Completed result: neither candidate is promoted

| Arm | Correct /100 | Paired gains | Paired losses |
| --- | ---: | ---: | ---: |
| Current relative-reservation control | 81 | — | — |
| Fine user retrieval, 3,072 tokens | 72 | 2 | 11 |
| Fine user retrieval, 2,048 tokens | 71 | 5 | 15 |

The 3,072-token candidate recovers the follower comparison and plant count
(50 and 53). It loses 7, 13, 27, 45, 52, 58, 65, 77, 81, 89 and 93. The
smaller candidate gains 34, 42, 50, 53 and 80, but loses 7, 13, 16, 27, 30,
35, 40, 45, 52, 58, 77, 81, 89, 93 and 95. The current retrieval path remains
the control; do not replace it with either global fine-ranking variant.

All 300 fresh Terra responses sealed before 158 physical Sol judgments for
the 300 logical answer rows. Reader and judge replay reproduced the reports
with 300 and 158 checkpoint hits, respectively, and zero new provider calls.
Both execution and replay exited successfully.

| Artifact | SHA-256 |
| --- | --- |
| answers.json | `cf0aac7ba381e01c7e9d382b7ace4f4498a28813bc3d95c7af5b9b1bdac18cd8` |
| report.json | `ce10f7637c434edefd4d19a007184076284869417ec697ece7f8c1895128bd17` |
| complete.json | `e0adb2e3a0e046e6742d4444ecde80c69a802c6fa3617a7ee413fc20344cd188` |
| paired-outcomes.json | `adb2a0d5fdbe1ea45645be30238b4dcda90c236df08b0eb8578b6a6b581ae423` |
| annotated-coverage-diagnostic.json | `282362085fbc88ea7bdedc83fa5582bd88967e96bbe89ad07c7dfb594842357f` |

The post-answer coverage diagnostic reuses the bound native-history annotations
strictly for analysis. Of 97 questions with annotated turns, complete annotated
coverage is 84 for the control, 82 for fine retrieval and 76 for compact fine
retrieval. Both fine variants recover the missing plant and clothing statements.
The clothing answers remain wrong despite that coverage. Other supporting
turns are displaced. Annotation coverage does not establish semantic sufficiency
and must never select production sources or route individual questions.

The 81/100 control is a new batched response population with temperature zero.
It cannot establish a temperature improvement over the earlier 78/100 batch,
or replace the earlier independently timed 80/100 result. The joint accuracy
and latency goal remains open.

## Next boundary

Preserve the previous retrieval path and these frozen failures. The fine index
may support an additive or source-scoped experiment; its global replacement
has now failed. Do not keep tuning global fine-ranking budgets against these
outcomes or restart the completed jobs. The smaller packet does not justify a
serving-latency campaign while its answer quality is this low.

A useful next reader experiment is to retain every currently selected raw
span while presenting related excerpts together under opaque conversation
labels in original transcript order. The current renderer interleaves sources
and repeats timestamps; grouping can test whether local conversational context
and less framing improve interpretation without removing evidence. Exact
span identities and raw bytes must be preserved. This is a proposed next
experiment, not an implemented improvement or a measured result.
