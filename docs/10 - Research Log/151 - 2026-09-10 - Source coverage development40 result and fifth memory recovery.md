# Source coverage development40 result and fifth memory recovery

**Date:** 2026-09-10  
**Status:** comparison complete; full100 target unmet  
**Predecessor:** [150 - Source diverse coverage and three way routing comparison](150%20-%202026-09-10%20-%20Source%20diverse%20coverage%20and%20three%20way%20routing%20comparison.md)

## Fresh matched result

The frozen three-way comparison completed all 280 fresh Terra requests across
40 questions and four complete approximately 1.04M-token memories. Reader v2,
the 3,072-token evidence cap and serial counterbalanced API controls were held
fixed. The 120 logical Sol judgments required 59 physical calls (15, 12, 15,
17 by namespace), then replayed identically with zero new calls. Session 68509
exited successfully. No model, prompt or routing change occurred during timing.

| Memory offset | Facet control | Scoped terms | Source diversity and scoped terms |
| --- | ---: | ---: | ---: |
| 0 | 10/10 | 8/10 | 9/10 |
| 10 | 8/10 | 8/10 | 9/10 |
| 20 | 10/10 | 10/10 | 9/10 |
| 30 | 6/10 | 7/10 | 8/10 |
| Total | 34/40 | 33/40 | 35/40 |

| Method | Memory median/p95 total | Identical-evidence API median/p95 |
| --- | ---: | ---: |
| Facet control | 6.033 / 9.634 s | 5.418 / 7.688 s |
| Scoped terms | 5.980 / 9.086 s | 5.651 / 8.297 s |
| Source diversity | 5.914 / 11.791 s | 5.267 / 7.899 s |

The shared short-API baseline is 4.741 / 5.686 s median/p95 total. Median local
preparation is 0.259, 0.274 and 0.303 s respectively. Visible first-content
time is almost complete-response time at this gateway; these observations do
not establish earlier token streaming. All three methods fail the provisional
joint latency allowance. Source diversity has five misses already; an unchanged
full100 run would have to answer every remaining question correctly to reach
95/100. The other two methods cannot reach 95/100 unchanged. No method is promoted.

Aggregate report:
`eval_results/full1m-spine-source-coverage-development40-20260910-r1/development-report.json`,
SHA `45a9f03703182a1672348268902068db0472ce1c865dda486b75457de835564f`.
The final namespace report SHA is
`3a8d0c8288429a1cfb956153d91687c7f8408b7d03b821e3f4559b3ba49a433a`.
The aggregate binds all four independently sealed answer/judgment populations.

## Postscore diagnosis

Source diversity gains ordinals 14, 31 and 38 over the facet control and loses
5 and 27. The recovered raw purchase now produces **70 pounds**, and the named
Sunday schedule produces **8 am–4 pm**. Evidence recovery helped these answers,
but broader packets reduced recommendation precision elsewhere.

Its remaining misses are 5, 13, 27, 34 and 36. These identifiers are diagnostic
only and must never select production routes, prompt variants or stored answers.

- Camera recommendations combine genuine raw mentions of multiple camera
  systems, losing the expected current compatibility preference. All inspected
  camera mentions precede the question; a hard timestamp cutoff would not fix
  this case. The latest Sony setup is May 27, after the May 26 Nikon mention.
- The table-tennis answer incorrectly uses tennis frequency.
- The painting packet contains the challenge, Instagram and tutorials, but
  its concise recommendation omits specific preference bindings.
- The delivery answer counts four named providers despite a question asking
  for types. The packet includes pizza delivery, an aggregator and prepared
  meals; category-versus-brand counting requires precise qualification.
- The entertainment packet lacks the relevant stand-up comedy preference.
  Its recommendation instead follows retrieved scripted-series interests.

No new prompt or routing revision has been selected from these observations.
The previous reader-v3 result is a separate run and its correct answers must
not be merged into this one. Likewise, the historical 95/100 used a cumulative
retrieval/repair lineage over the same roughly 1M-token scale, without a fresh
joint demonstration of API-like latency; see Research Log 132.

## Fifth-memory compaction recovery

The original first compaction response was preserved. The new completion tool
executes only the one unstarted original batch, with no resend of completed or
unacknowledged requests. Its completed inventory identified seven valid
summaries and two invalid slots across the nine jobs.

Completion root:
`eval_results/full1m-spine-original-compaction-completion-offset040-20260910-r1`.
Preflight SHA `f1ea95ddc07d063fcb6be135359daecec3de9532e673324047fb17e18c4d6374`;
complete SHA `e7ed9ca17d8e3a728a66299d94ee0b42d43e2c14b1264cd6b354230e56c93b3a`.
One new original-batch call and one authenticated completed-batch hit occurred.

Recovery root:
`eval_results/full1m-spine-budget-recovery-offset040-20260910-r1`.
Preflight SHA `97d9ee53176bd4a9ed3c2dc6817b59632c602944b74d695f5171c4207bd67502`;
repairs SHA `717cecc5bf6b626b8b3193637c3f149750db0598194db8048f1ffb44b6a2d46a`.
Two calls recovered the two invalid summaries within the maximum four-call
allowance. Seven previously valid summaries remain unchanged. Qwen inputs
contain summaries only. Admission replays both original and both recovery calls.

`tools/admit_spine_corpus_v6.py` adds explicit authenticated recovery dispatch
under source-binding policy v9, preserving older admission implementations.
The version-8 conditional verifier normalizes the equivalent old/new rules and
replays the complete attributed recovery and raw partition. The full100 v2
report retains the existing accuracy and latency gates with this verifier.
All **147 focused regression checks pass**, including unknown-request refusal,
exhausted-recovery refusal, complete source admission and the full100 gates.
The initial checks caught a dictionary-iteration error and missing recovery
dispatch; both were corrected before real recovery/admission execution.

## Remaining execution

Fifth-memory admission, verification and all indexes are complete. Its 5,492
fragments form 2,737 leaves across 481 sources and 1,046,567 token proxies.
The exchange merge required one original Qwen batch and six bounded individual
recoveries; leaf merges then completed in six batches, for 13 Qwen compilation
calls beyond the four source-summary compaction calls. Local Qwen attention
received user summaries only, and parent summaries remain deferred.

| Artifact | SHA |
| --- | --- |
| Source atoms | `c08d92638bcedbd3ad32a4d58da3300c8a3479db0d6979a7025974acefe0f064` |
| Admission v8 verification | `e03084105a3194340eac99223d5a1038c97062b9ed1e511f0650a880ea0f149f` |
| Attention leaf hierarchy | `c41b96ec021f537c1a0f1dd22d05ce79ab84045d9034322bf587ba1c8cb57639` |
| Semantic index | `6f62d12c319630cda1766607f1fc597f15c05f9e311fc6a9899bc22e014ce7b0` |
| User-summary addresses | `d6c08f6d8ae97a07a98c0c3c87622f04cd439c53ac84561e73249503e1e93601` |
| Passage addresses (5,783) | `c3eb5c8220b598062c0b439057223467c13ce68a6918ef910c11d4fbd9c059c4` |

All five complete memories share admission-method SHA
`92187190058445ed4fdbd8375fa870162117fa1b1b913f2aeb2b7c9bc2bcabcf`.
The four older memories replay with unchanged atoms and zero provider calls;
their aggregate receipt is `admission-four-memories-v8.json` under the completed
development40 root, SHA
`617bb12d74b6ab0c3656bd1cf2b32300b3d927b36b615dd93fc8627812fe3b57`.

The comedy source is intact and ranks sixth in the whole-summary frontier, but
is absent from the user-summary and passage top32 frontiers. A new experimental
`spine_combined_memory.py` retains all prior users, adds the first user-containing
leaf from at most eight distinct sources in the whole-summary top32, then
performs the same exact hydration. It reuses the existing query vector and
source-coverage checks. The fixed population diagnostic in
`audit_spine_combined_coverage.py` compares all 40 prior control prompts before
publishing candidate prompts; no predictions, references or witness IDs enter
selection. The audit completed with **40 byte-identical control prompts**, 37
changed candidate prompts, 112 added spans, 27 displaced non-user spans and
**zero removed user spans**. It uses the same 3,072-token allowance. Its aggregate
is `eval_results/full1m-spine-combined-coverage-development40-20260910-r1/audit.json`,
SHA `fe87b01345c54313c0b4d9619ba0eb503be583a8fc5a97e08ccfb6ee9d391cba`.
The offset-0/10/20/30 audit SHAs are respectively:

- `ecfa1f381f6f7429b87244aab5c12248061c65bda7aa7ef9c5e783c6ea7e5bc0`
- `a2043a15972fc062c4464c14bac621ebaf5a94f3ef5e1acfe9c111674c8493d9`
- `914b0c1369c77bb29f3e3fcb1ecea0b166aaf9940e7d135563a10c4ddd8d821d`
- `e85671633e628cb55c54a3435c4666173c7b64ca4be9be5f0081e45445911b2a`

After production selection completed, inspection confirmed the actual comedian's
Netflix request in ordinal 36, candidate-prompt SHA
`d171d48c6d82ce882ee05319ca68c7827e20b42c155c492198968537ce45cd9d`.
This is evidence recovery, not measured answer accuracy. No successor answer
experiment has been prepared or released, and no new reader policy was chosen.
Further evaluation should include the now-ready fifth memory rather than keep
testing only the same four development memories.

Sixth-memory raw ingestion is running in session 65309 from the already
prepared 834-request namespace at offset 50. Readiness passed for both models;
report `eval_results/spine-gateway-readiness-offset050-20260910-r1/report.json`,
SHA `1b6085207d990a524e8d876665cc159ac9bd0da7a265b0854c79bd9cb6b3f003`.
Do not start timed answer evaluation alongside bulk ingestion or GPU compilation.
Namespaces 60 through 90 remain prepared and unstarted. Confirmation200 remains
unopened. The active full100 joint accuracy/latency goal is not achieved.

The README was accidentally truncated during line-ending normalization and
recovered before this handoff from saved document captures, prior repository
content and recorded edits. Captured initial prefix/tail, latest prefix, original
navigation line and the 447-line updated document were checked before writing
the recovered file. Recovery evidence is retained under the development40
root's `readme-recovery/`; the Git index was unchanged. The subsequent fifth-memory
status update is intentional. `git diff --check` passes for the updated README
and tracked test files.
