# User statements before assistant context

Log 230 completed a 100-question answer comparison after an evidence-preserving
lexical addition. Coverage reached 98/100, but accuracy fell to 87/100. Seven of
eight regressions had exactly the same prompts as the prior 93/100 run. The next
work therefore addresses the reader rather than changing retrieval budgets again.

## Alternative answer-model availability

`tools/probe_native_spine_answer_model.py` locks four evenly spaced existing
question ordinals (0, 25, 50, 75) for transport/latency readiness only. It does not
open references, grade answers, ingest history or call Qwen. The admitted local
gateway's `claude-sonnet-5` route rejected the first request with HTTP 400:
the upstream Anthropic account has insufficient credit. There are zero successful
responses and no latency or accuracy result. No other Anthropic model was probed.

- Root: `eval_results/native-spine-sonnet-readiness-20260915-r1`
- Probe process: terminal exit 1.
- Failure receipt: `7f54a3b4636b665c9539d37f29fad3092c33e31963cff5f0555c6e7ab09e1cdd`
- This blocks that answer-model comparison, not progress with the available Terra model.

## Layout change with identical evidence

The existing v7 reader already explicitly instructs completeness and speaker
attribution. The new `application/user_spine_section_context.py` instead changes
the arrangement of the exact evidence. Within each selected conversation,
`USER_STATEMENTS` contains all selected user turns in transcript order;
`OTHER_TURNS` then contains the other selected roles in transcript order. The
original chronological T labels remain on both blocks. Original conversation
ordering, timestamps, text bytes and exact span coverage are preserved.

No selected evidence is pruned, summarized, duplicated or newly retrieved. Raw
text is never parsed for role tags, so literal markup in user text remains raw
text. Adjacent slices from the same turn can merge; gaps cannot. Both the flat
hydration and rendered packet remain within the original 2,048-token cap.
The Qwen hierarchy still comes from user-spine summaries, with no query-time
Qwen pass. This change is presentation, not raw-text attention.

The old renderer, application implementation and evaluation files remain sealed
and unchanged. `tools/native_spine_user_spine_presentation.py` binds the new
renderer to the existing retrieval and independent raw-audit boundaries.

Eleven renderer checks passed in 1.32 s. Sixteen evaluator, renderer and
application-gate checks passed in 3.04 s. The initially failing budget-test
fixture had already exceeded the flat hydration budget; it was corrected to a
valid one-turn fixture that actually exercises the new framing limit.

## Complete provider-free preflight

`tools/assess_native_spine_user_layout.py` reconstructs the original source order
from the body bank and verifies every rendered byte. It opens no references and
makes no model calls. All 100 routing/hydration receipts are unchanged from the
87/100 additive run; all 1,439 selected raw spans remain present. Median rendered
length is 1,552.5 tokens and the maximum is 1,894, below the unchanged 2,048 cap.

- Root: `eval_results/native-spine-user-layout-preflight-20260915-r1`
- Exec session: `20587`, terminal exit 0.
- Report: `faa8485e87f9ade430d3441da7506bfbb1797c0e473177421ac02de1e8487e8e`
- No new ingestion, Qwen compilation or answer calls.

## Completed full 100-question run: 93/100

`tools/evaluate_native_spine_user_layout100.py` uses the actual reopened
`AdditiveLexicalMemoryCondenser`, the same v7 reader, Terra answer model, numeric
retrieval policy, original 100 questions/references and Sol grading. Preflight
requires all 100 live routing/hydration receipts to equal the baseline, and all
renderings to equal the independently audited layout packets. Only presentation
may change. The timed memory arm freshly embeds/retrieves/renders each query;
the control sends the identical prompt directly to the same answer model.

- Root: `eval_results/native-spine-app-userlayout100-20260915-r1`
- Log: `eval_results/native-spine-app-userlayout100-20260915-r1.log`
- Exec session: `71180`, terminal exit 0. Do not duplicate the run.
- One persisted history: 1,098,417 raw tokens and 5,357 raw turns.
- 100 memory answers plus 100 alternating matched API controls, then 100 grades
  and independent raw reconstruction. No references enter retrieval or answering.

The completed score is **93/100**, versus 87/100 for the immediate layout
baseline. It ties the previous best rather than achieving 95%. All 200 answers
stopped normally and reported the expected Terra alias. All 100 memory packets
and 1,439 original raw spans passed independent audit. A provider-free judge
replay (`9846`, terminal exit 0) returned 100 cache hits, zero new calls and the
identical report. No original grade was changed.

| Measurement | Memory | Identical-prompt API |
| --- | ---: | ---: |
| Warm median total | 4.651 s | 4.581 s |
| Warm p95 total | 7.084 s | 7.170 s |
| Answers below five seconds | 62/100 | 62/100 |
| Mean total | 4.964 s | 4.794 s |

Median preparation was 0.272 s. Cold setup was 25.376 s and is excluded from warm
latency. The memory/API median ratio was 1.015. The same 1,098,417-token memory
was reused throughout; there was no reingestion or Qwen compilation.

| Artifact | SHA-256 |
| --- | --- |
| Preflight | `8f35949288c55337d62837e26b04fab5ced8588b77591f4848bfb968c2cf3638` |
| Joint report | `61edbb6f3fd4202d93236d92f40d5628375fe1d6de02b1c27a7bc20a1bceafde` |
| Raw audit | `7963aadbf58216287fb11a36abe3b15404ade25bf19d3204db1b99467e921af0` |
| Comparison | `0f7c66e4110894980cc69760b17bbfcf9d010a3fa410dc94768865b7d7712353` |
| Failure diagnosis | `87149fe35e26201700f74602022888d980f75d1f8c222a22633e10aea1fdbd58` |

These are exposed generated questions over real transcripts, not an official
LongMemEval or generalization result. A single six-point gain over the immediate
baseline does not establish repeatability in the presence of the answer/grading
variation already observed. The 95% target remains unmet.

## Seven misses and two concrete routing gaps

Compared with the immediate 87/100 baseline, gains are 6, 14, 19, 38, 65, 69 and
98; the loss is 26. Compared with the earlier best 93/100, gains are 19, 35 and
38, with losses at 81, 95 and 96. The remaining misses are 26, 53, 81, 82, 93,
95 and 96. Do not combine correct answers across runs into a fabricated score.

The main reader failures are the omitted nightmare clause in the supplied story
opening (81), and mixing the online-shopping cashback conversation into the
household-shopping question (96). The extra festival resources at 26 and French
cinema interests at 95 are real user statements rejected for absence from the
gold answer. At 82 the answer covers all requested drama requirements, but the
judge additionally demands unasked show/character names. These diagnoses leave
the original seven incorrect grades intact.

The questions at 53 and 93 are ambiguously scoped among multiple real user
statements, but their specific reference support is also only partially routed.
Tracing the exact original source addresses gives actionable causes:

- **53 (concerts):** The reference conversation's Imagine Dragons atom ranks
  eighth among direct matches. It is hydrated, but its source gets no parent
  expansion because only the first four direct matches seed parents. The
  Governors Ball and Ariana Grande user statements are absent from the route,
  not dropped during hydration. Flat usage is 1,805 of 2,048 tokens.
- **93 (collections):** The reference source has a rank-two direct match and a
  selected parent. That parent's user spine covers camera and sci-fi discussion
  but begins after the vinyl statement. The vinyl atom is absent from the route.
  Flat usage is 1,691 of 2,048 tokens.

Both omitted statements survive in the stored broader parent user-spine
summaries. The diagnosis receipt binds their original support quotes, candidate
packets, direct ranks, consulted parents and available parent summaries. This
is a hierarchy/source-expansion gap with spare raw budget, not a need to rerun
ingestion or let Qwen read raw text.

The next bounded experiment should use the stored broader parent **user-spine
summaries** to select supplemental exact user atoms. Parent-level semantic
ranking can distinguish a conversation that jointly covers the question from
individual high-scoring atoms in other conversations. Preserve the existing
route prefix and append bounded additions so no original evidence is displaced.
Check all 100 packets and original-source coverage before another full answer
run. Do not add source IDs, question IDs, gold text, or example-specific words
to routing. Keep all existing numerical/raw limits and disclose any new summary
index compilation separately from ingestion. This plan is not yet implemented.

## Timestamp handling beyond this native population

The native router requires one timestamp per source occurrence. An additional
check confirmed that invariant on every served conversation in all 100 packets;
its receipt is `b34f7583e785f4585a48f0d5ded20848d1d3b511f25f8444ddb29ee0d1d1be63`.

Review found a more general renderer edge case: after a later user turn, switching
to an earlier assistant turn must explicitly restore the earlier timestamp.
The sealed v1 renderer reset internal timestamp state at the role-block boundary
without always printing that restoration. The benchmark cannot encounter this
case because of the native occurrence invariant, but general callers can.

`application/user_spine_section_context_v2.py` fixes that state carry while
preserving the historical v1 file. Four focused checks passed in 1.30 s, including
the backward timestamp switch. An independent reconstruction of all 100 benchmark
packets proves v2 produces identical text, placements, token counts and metadata,
except for its explicit format version. No new answers were generated and no
new accuracy score is claimed. Use v2 in the next implementation.

- Equivalence root: `eval_results/native-spine-user-layout-v2-equivalence-20260915-r1`
- Report: `84e1259f88f2df63400f7ffae1b91b482d82ad9210ebf9eaf6abe44d3249fa69`
- Historical full-run evaluator still binds v1; do not edit its sealed dependencies.
