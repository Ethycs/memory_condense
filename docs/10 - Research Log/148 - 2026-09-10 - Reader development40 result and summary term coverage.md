# Reader development40 result and summary term coverage

**Date:** 2026-09-10  
**Status:** 33/40 versus 32/40; neither unchanged reader can reach the full100 target  
**Predecessor:** [147 - Preserved recovery execution and resumed reader comparison](147%20-%202026-09-10%20-%20Preserved%20recovery%20execution%20and%20resumed%20reader%20comparison.md)

The prepared reader-v3 comparison completed all 240 answer requests across
four complete approximately 1.04M-token memories. Independent Sol judging
scored the revised reader **33/40 (82.5%)**, versus **32/40 (80%)** for its
fresh reader-v2 control on identical raw evidence. Seven misses already limit
the revised reader to at most 93/100 if these predictions are retained; the
control's corresponding ceiling is 92/100. Neither version is promoted as a
solution to the joint target.

The earlier 95/100 remains the cumulative retrieval and answer-repair result
over approximately 1M-token memories. It does not establish API-like latency
for this new method. The new comparison remains development evidence, and
confirmation200 stays unopened.

## Accuracy and exact execution

| Memory offset | Control | Revised reader |
| --- | ---: | ---: |
| 0 | 10/10 | 8/10 |
| 10 | 7/10 | 9/10 |
| 20 | 9/10 | 10/10 |
| 30 | 6/10 | 6/10 |
| Combined | 32/40 | 33/40 |

There are 30 shared correct answers, three revision gains (ordinals 13, 14,
27), two revision losses (5, 6), and five shared misses (16, 31, 34, 36, 38).
The revision improves table-tennis qualification, cuisine counting and the
painting recommendation in this run. It loses the photography recommendation
and descriptive artist identification. The same evidence was used by both
readers throughout; outcomes were never used to choose per-question methods.

All 240 Terra requests completed with no answer transport failures or retries.
The 80 logical Sol judgments required **61 physical calls** (14, 17, 16, 14);
identical judge prompts share authenticated completions. All four judge
populations then replayed with the same 61 hits, zero calls and identical
reports. The serial runner, session **62889**, exited successfully before
any new bulk ingest began.

Root: `eval_results/full1m-spine-reader-v3-development40-20260910-r1`.

- Combined report: `b015616f74f78f99d3bf5ed7d6880cc90f1a61c765b1f62efaed84886cc7e4a7`.
- Diagnostic: `ca48664236c7351039b42f21c38e50a77402602dc462e07f18fb61411b9f9ea5`.
- Offset 0 report: `43adc59dfe5699fa3b7a81a852c4225df07a5ceefdca043470a1c133f778ee86`.
- Offset 10: `8c50f0ec6a031f521b6f63b28cf5e2292e00c48c9aa214546b94ac26f70c73f4`.
- Offset 20: `f5cda76c4381d8076fefcfd04fd714f4a84e2c90d4ef4d267f35668bf0c59eb4`.
- Offset 30: `7df739f41ac6a69cd3a49097139de51d0046e1b222e247eb3fe8e6e6b87d4897`.

The diagnostic reauthenticates all answer/judgment bindings and identical
evidence pairs. Its full100 ceilings describe these fixed predictions, not a
future rerun or an estimated score for unasked questions.

## Joint latency

| Reader and arm | Median total | p95 total |
| --- | ---: | ---: |
| Control memory | 5.637 s | 11.488 s |
| Control identical-evidence API | 5.606 s | 9.688 s |
| Control short API | 4.513 s | 5.423 s |
| Revised memory | 5.790 s | 9.839 s |
| Revised identical-evidence API | 5.743 s | 9.636 s |
| Revised short API | 4.616 s | 6.853 s |

The revised memory/API ratios are **1.008 median and 1.021 p95**. Against
short API chat they are **1.254 and 1.436**, exceeding the provisional 1.10
allowance. Visible TTFT is essentially total response time in these gateway
streams and reaches the same pass/fail conclusions. Revised routing and
hydration take 0.331 s median and 0.393 s p95. The control also fails latency,
including its identical-evidence p95 ratio of 1.186.

These are the same fresh streamed answers that were judged for accuracy.
No bulk ingest, GPU compilation or large replay overlapped the timed run.
This does not pass the joint target: neither reader reaches the accuracy
requirement, and both exceed the short-API latency allowance.

## New failure evidence and experimental supplement

The fourth memory adds four shared misses: a feed-purchase total, a count of
delivery services, a media recommendation and a named shift assignment.
The earlier shared apartment-duration miss also persists. These require
separate evidence checks; a wrong final answer alone does not locate a failure.

The shift-assignment packet demonstrably omits the required named exchange.
It contains the opening request, several constraints and early generic agent
tables, plus unrelated user turns. The stored hierarchy nevertheless contains
a leaf whose user summary lists the supplied agent names and whose paired
assistant summary describes the resulting Sunday–Saturday named schedule.
This is a selection/hydration gap, not absent source data.

`search/spine_term_coverage.py` adds an experimental summary-only supplement.
It considers the two rarest present query terms occurring in at most 1% of
leaf summaries, offers up to two matching exchanges per term, preserves the
existing user-section prefix, and places those exchanges' paired context
before the previous assistant-context tail. It contains no benchmark names,
question IDs, answers or score-dependent selection. The original Qwen-compiled
attention hierarchy remains unchanged, and the new router reads no raw text.
The original exact hydrator still enforces 3,072 tokens and 128 raw spans.

Nine focused checks pass, including rare-name examples unrelated to the
benchmark, exact context recovery, preserved user evidence, no-op common or
absent terms, and rejection of foreign queries or altered source descriptors.
The existing source expansion checks pass in the same run.

`tools/audit_spine_term_coverage.py` completed all four memories,
under `eval_results/full1m-spine-term-coverage-offsetNNN-20260910-r1`.
It must reproduce every frozen baseline prompt, preserve every previously
hydrated user section, and account for all added/displaced raw spans. It reads
no reference labels or predictions and makes no answer calls. This is a
retrieval diagnostic, not new accuracy evidence; do not promote it from source
coverage alone.

The complete diagnostic reproduced all 40 baseline prompts and preserved all
prior user evidence. It changed 32 prompts, added 114 spans and displaced 51
non-user spans. The named schedule was recovered exactly. However, corpus
rarity alone also selected conversational words such as "remind" and
"wondering", admitting unrelated reminder conversations. This unscoped version
is not promoted. Its aggregate is
`term-coverage-four-memories.json` under the reader development40 root,
SHA `6c1352c2beffd9840163ebaa0e266b3ac0e4bd87ea8b437ea174ce818503ae70`.

The successor `search/spine_term_coverage_v2.py` requires the source
conversation to have been selected by the existing summary router before a
rare term may refine its exchange selection. It retains complete-memory
document frequencies rather than redefining rarity within the selected source.
It introduces no new source and adds no query-time model call. Eleven focused
checks pass, including the original nine and source-scope/non-mutation checks.

`tools/audit_spine_term_coverage_v2.py` also completed all 40 questions,
preserving every baseline user section and the exact original prompt controls.
It changes **27 prompts**, adds **46 spans** and displaces **22 non-user spans**;
all additions belong to previously selected conversations. The named schedule
still appears with its exact Sunday row and Day Shift header. Neither
supplement has fresh answer judgments or latency measurements.

- Scoped aggregate: `b1c436e8405ea8651e683fb28b65c84639772540e05eb531eed604006b9545b3`.
- Offset 0 audit: `a9b66899e7df10c1fdeb3fa72a37461504f7df4c003d5e8f780b34f667e2fb05`.
- Offset 10: `3d4dd18330779e599f7f132263f4340593e5ca7fb99c01cc4188c258a6db0724`.
- Offset 20: `7513e7c5c4931bc853728cf9fec995adfebb2c9d74bc155791b5066566fea573`.
- Offset 30: `c494019cce6b6ad8d9f4fe8d1b23a808625bcdb2f413633ba441633ff7feba36`.
- Named-schedule diagnostic prompt: `793ddb5600a36ab133fd12685e31ca4740ee8d9ec25a7ecc7d06cb68ad4876bf`.

The scoped audit roots are
`eval_results/full1m-spine-term-coverage-v2-offsetNNN-20260910-r1`.
The aggregate is `term-coverage-v2-four-memories.json` under the reader
development40 root. Both completed diagnostic implementations must remain
reproducible; do not overwrite them to prepare another method.

## Timestamp diagnostic and its limit

A separate metadata audit found **168 of 905 rendered excerpts** later than
the supplied question timestamp, across **32/40 packets**. The shift question
alone contains ten such user excerpts. This may consume useful context, but
it does not establish that removing those excerpts improves accuracy.

Existing Logs 133/134 already record required benchmark support from a session
one hour after its question timestamp. A strict timestamp cutoff would discard
that support. No hard cutoff was implemented or promoted. The audit remains a
diagnostic about the existing rendered timestamps, not a causal availability
guarantee or a substitute for event-date interpretation.

Artifact: `question-availability-diagnostic.json` under the reader development40
root, SHA `41ac06dac6ed621e55a39af908027c95eca7c2f3fed8d60bdc59dd6eb757380a`.

## Resumed full-corpus construction

Both fresh synthetic readiness probes passed at **09:26:19 and 09:26:22 UTC**.
Report: `eval_results/spine-gateway-readiness-offset040-20260910-r4/report.json`,
SHA `ca52f5c6ef6f30cb132b7724867127e3478cabaa4b6c85add3760b98ee136dcc`.
The prepared v2 executor then started offset-40 recovery in session **43809**,
using its frozen 738-call budget: 733 first attempts plus five bounded
additional attempts, while retaining the 99 prior responses. Original unknown
reservations remain untouched. No timed answer evaluation runs alongside it.

Poll that live session rather than starting a duplicate. On normal completion,
verify the full transport lineage, run the complete source-admission audit,
compact only oversized stored summaries with Qwen, and reproduce source atoms
under version-7 admission before constructing the fifth memory's hierarchy and
addresses. Raw validation diagnostics are not final source-admission results.
Offsets 50 through 90 remain prepared and unstarted. The next retrieval
comparison must measure the scoped supplement's fresh answers and latency on
complete memories, keep its reader fixed, and account for any displaced
assistant evidence. No answer preflight or provider budget for that comparison
has been executed yet. Source recovery and the remaining complete namespaces
continue independently of reader-policy promotion. The full100 objective is
active and unmet.
