# Relative day user evidence reservation

The user asked what the agent was doing after a prolonged corpus-preparation
detour. The new native corpus is parked. This experiment returns to the existing
complete memories and their 80/100 as-of baseline, making one bounded routing
change rather than compiling another corpus.

## Concrete failure and change

For “What kitchen appliance did I buy 10 days ago?”, the old packet omits the
user's statement that they got a smoker. The statement is present in the raw
memory and accurately represented in its user-spine summary. Enabling the old
extended relative-date hint still omits it: that method limits the preferred
frontier to four sources from twelve eligible sessions on the indicated day.

The additive `RelativeSpineReservation` retains one summary-ranked user turn
per source from a single explicit relative-day window, capped at twelve sources
and a 1,536-token metadata reservation. Those whole user turns precede the
existing as-of plan; exact duplicate sections are removed. The final hydrator
still enforces 3,072 context tokens and 128 raw spans. Mention dates remain a
retrieval hint, not proof of event dates. Other evidence remains available in
the original plan, subject to the final budget.

Selection reads summary scores and authenticated span metadata only. Qwen still
receives no raw content. No new Qwen calls, embeddings of raw text, benchmark
identity routing rules, reference-based source filters, or reader changes are
introduced. The original frozen runtime remains unchanged.

## Verification and real-data evaluation

Twenty-nine routing/hydration checks and seven evaluation checks pass. They
cover exact bytes, future and assistant exclusion from the reservation,
source diversity beyond four sessions, budgets, no-op queries, foreign
coordinates, complete populations and the answer-before-reference barrier.

The ten-question smoke check includes the missing smoker statement at 3,026
context tokens and leaves the other nine packets byte-identical. Full100
preparation changes only ordinals 43, 54 and 93. The fresh accuracy screen
compares both methods across all 100 questions, including prior successes.
Its 200 logical requests require 103 physical Terra responses: identical
prompts across arms share a single fresh response. No old predictions are
reused. All answers must be sealed before Sol judging.

This is a batched accuracy screen, not a serving-latency measurement. It cannot
pass the joint target or inherit the old latency result. Repeated accuracy on
unchanged prompts can differ from the previous 80/100 run; the relevant routing
comparison is the fresh paired control, with shared answers for identical
prompts. Any candidate promotion still needs the joint full100 latency gate.

Root: `eval_results/full1m-spine-relative-reservation-full100-20260910-r1`.
Preflight SHA:
`2ee6fb269c73bb2b1653460148d3b840065334b37dd06562710e0cc921b4a077`.

The old extended-date diagnostic completed separately at
`eval_results/full1m-spine-relative-offset050-20260910-r1`, audit SHA
`8b8e7afa867a9b94a2bdc8e4394928b7918f997413ddc996c4cebc83cb982faa`.
It does not contain an answer-accuracy improvement.

## Completed result

The fresh control scores **76/100**, and relative reservation scores **78/100**.
There are two paired gains (54 and 93), no losses, and no change in correctness
for the third modified packet (43). The smoker answer replaces “I don't know.”
The business-milestone answer changes from an influencer collaboration to
signing the first client contract. Both candidate answers are grounded in exact
user statements on the respective indicated days.

All 103 fresh Terra responses were sealed before 103 physical Sol judgments.
Both arms have 100 fresh logical answers; 97 unchanged prompts share responses
across arms. The 200 logical judgments also share byte-identical judge prompts.
Offline replay reproduces both scores with 103 reader and 103 judge checkpoint
hits and zero new calls. Execution and replay sessions exited successfully.

| Artifact | SHA-256 |
| --- | --- |
| answers.json | `94795dbd118e85481b2dbb9047af02c58c746a66571228706df272a7ac2b14e4` |
| report.json | `e8455fb676b2f0b0e726082ff664a1dd0cf495cb7cc544d2a0b1b7c0bc132cbc` |
| complete.json | `a52c7a2ed34336de4899e7a4cc283ec621618d5440d59e32747092aa6908502b` |

The new control's 76/100 differs from the old streamed 80/100 despite identical
evidence prompts: one old miss becomes accepted (60), while five old successes
become misses (5, 14, 77, 88, 93). Fresh generation, judging, transport and
concurrency differ; this run does not isolate which caused those changes.
Do not call the candidate 82/100 by adding its two gains to the old score.
Its observed score is 78/100 on this fresh batch, and the paired routing gain
is two questions. It is a narrow improvement, not a solution to the 95% target.

The successor code and sealed producers are now frozen. Keep the corpus detour
parked. Continue from the current packets and remaining failures; neither a
large new ingest nor another accuracy-only result is a joint target pass.
The last independently timed full100 remains 80/100, and the active goal is open.
