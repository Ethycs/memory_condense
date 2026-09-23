# Real data user spine routing evaluation

**Status**: HISTORICAL RETRIEVAL PILOT — answer/judge execution subsequently completed in Research Log 131
**Date**: 2026-09-09
**Applies to**: `perf/durable-ingest-pipeline`, `.worktrees/ingest-speed`
**Depends on**: [Research Log 129](129%20-%202026-09-09%20-%20User%20spine%20attention%20hierarchy%20and%20summary%20compilation.md)

**Successor**: The user's subsequent "keep going" instruction was accepted for
the prepared answer-and-judge scope. [Research Log 131](131%20-%202026-09-09%20-%20Real%20data%20user%20spine%20answer%20and%20judge%20results.md)
records all 40 predictions, judgments and zero-call replay. Approval-pending
and answer-unmeasured statements below describe this earlier stage.

## Measured result and limits

All five routers recovered the designated exact user-support anchor for all
seven source-derived development probes. Each missed some necessary evidence
on the separate real benchmark trip-chronology question. No answer predictions
or semantic judgments have run, so these are **retrieval coverage results**, not
answer accuracy. No candidate is promoted.

The source population is three complete conversations selected in the existing
r9 q86 packet's citation order: 39 turns, 18 user turns and 8,855 raw token
proxies. It is an already examined candidate pool from one question family,
not the full corpus or an untouched validation sample. Seven diagnostic
questions were written from those sources before routing. The original locked
benchmark q86 is reported separately; this is not eight benchmark questions.

| Router | Development support coverage | Qwen routing calls, eight queries | Qwen routing input tokens | Mean Qwen provider seconds/query | Mean raw context tokens |
| --- | ---: | ---: | ---: | ---: | ---: |
| BM25, both summary channels | 7/7 | 0 | 0 | — | 1,577 |
| Qwen, narrow exploration | 7/7 | 46 | 51,931 | 8.71 | 1,641 |
| Qwen, exploration beam 8 | 7/7 | 40 | 63,503 | 10.65 | 1,568 |
| BM25, user channel only | 7/7 | 0 | 0 | — | 1,569 |
| Qwen narrow, user channel only | 7/7 | 47 | 33,610 | 6.61 | 1,675 |

Provider seconds sum recorded completion-call durations. They exclude local
work, orchestration overhead, compilation, answering and judging. They are
single-run observations, not a latency distribution. Zero Qwen calls does not
mean zero BM25 runtime. Token counts use the repository's proxy tokenizer.

Removing attached-context summaries from Qwen routing reduced input by 35.3%
and observed provider time by 24.2%, but did not repair the chronology selection.
That follow-up was motivated by the first sealed run and is explicitly a
development ablation. The wider beam also failed to improve measured coverage.

All 40 selections have zero route errors and zero hydration diagnostics. Exact
support means a designated quote occurs inside a hydrated **user** span from
the correct source. It does not establish exhaustive recall or answer correctness.

## The remaining chronology failure

The dated question asks for the order of three past trips. Reading the selected
user evidence, without joining benchmark gold, shows:

| Router | Selected user-turn ordinals | Missing evidence |
| --- | --- | --- |
| Both BM25 variants | 7, 9 | Only the March source; Big Sur/Monterey and Yosemite trip evidence absent |
| Both narrow Qwen variants | 27, 18, 9 | April turn 18 concerns a planned July backpacking trip; it does not establish the Big Sur/Monterey trip |
| Qwen beam 8 | 29, 18, 14 | No Muir Woods trip evidence; May turn 29 says the Yosemite trip started today |

Source coverage alone would incorrectly make narrow Qwen look complete: it
selected all three sources, but the wrong April exchange. The underlying May
conversation also contains conflicting "got back today" and "started today"
statements. Transcript dates are mention times; the hierarchy must not silently
resolve that conflict. Generic top-k relevance remains insufficient for this
multi-event request. A future query decomposition/coverage experiment must
select supporting event assertions without hardcoding these trip names or
turn ordinals. There is no evidence here that attention hierarchy beats
conventional routing in answer quality.

## Compilation and implementation changes

Terra compiled the approved raw turns into 39 atomic summaries. Only 14 original
responses passed strict support checks. Removing external quotation wrappers
only when their interiors exactly matched source text, and splitting long exact
quotes without dropping characters, accepted 37. Two support-only Terra calls
repaired one more row; a final Terra label-selection call chose numbered raw
spans for the remaining row. **All original summary texts stayed unchanged.**
Exact supporting quotes verify provenance, not semantic faithfulness of every
generated claim. Support audit text never enters Qwen prompts.

The final builder reuses bounded singleton summaries and asks Qwen only for
merges. Channels increased from the original 64-token cap to an explicitly
sealed 128-token policy. Three earlier attempts stopped on excessive output:
9 original completions, one shortening completion, then 42 merge/shortening
completions. The final run reused the relevant 42 checkpoints and needed one
new completion. One attached-context compaction still had 139 tokens; the
final policy kept a 121-token complete-sentence prefix including an explicit
omission marker. User-spine omission is forbidden. This lossy summary operation
never truncates hydrated raw evidence.

The hierarchy contains 21 exchanges (18 user-led plus three preludes), 21 leaves,
39 total sections and three source-local attention windows. Nine exchanges
exceed the 512-token leaf target; the largest is 637 tokens and remains intact.
On this population all leaves are single exchanges, so this pilot does **not**
isolate a gain from attention-selected raw chunk boundaries. Attention changes
the parent hierarchy. The signal uses the pinned local Qwen3-8B six-layer prefix
over user summaries; the gateway supplies full Qwen summary reasoning.

Atomic summaries total 2,052 tokens; leaf routing summaries total 3,147; all
hierarchy summaries total 6,987. Raw evidence remains separately stored. This
is not a claim of total storage compression.

The added `spine_routing.py` keeps the exploration beam separate from the final
raw-section limit. The user-channel projection is an experimental evaluation
adapter; it preserves topology and raw pointers and does not regenerate
summaries. All arms use at most three routed sections, 4,096 hydrated context
tokens and 128 raw spans. Qwen receives summary strings and questions only.
No default production retrieval path changed.

## Sealed artifacts and replay

Paths below are relative to this worktree's `eval_results/`.

| Artifact | SHA-256 |
| --- | --- |
| `user-spine-hierarchy-real-source-20260909-r3/preflight.json` | `a261fc3d17e9b9d261f4239996078ea7941199534b24270b82c536d4489154e5` |
| Same root, `atoms.json` | `f82181a870947eeacbded82527541168fb51e382b98353e0e5326309a56a0d1f` |
| Same root, `hierarchy-build-preflight-r4.json` | `5244c17789056fa1cc2bc052d25aea687efa899a667c38a2ce84bfb5ac0a1581` |
| Same root, `hierarchy-r4.json` | `dd683c2639f4842df09f33c1ed2395a6e4f4a021b3256f5f24fa033a885b3dc5` |
| `user-spine-real-matched-pilot-20260909-r1/selection.json` | `8fa402d2cc188cfbb47b14e8b3a00f5d48b39cc3263c6298ae0ad5462035ce71` |
| Same root, `routing-evidence-audit.json` | `cc4b78e18700e165c497ebe0eee54b0b7215effdffbc109ad50ba5eb1d5fb9a3` |
| `user-spine-real-projection-pilot-20260909-r1/selection.json` | `e4f61736a8c1bd0b0d126c665e99efc6030d2d816342f6cbd397c588482b5dbc` |
| Same root, `routing-evidence-audit.json` | `f9d2d9f516a758bae04f3fe5c5418057f5f9663fb0cfe377101485e790d2f0a9` |

The final hierarchy replay authenticated 43 Qwen hits with zero new calls and
the identical hierarchy SHA. Original raw support preparation replayed all 39
Terra completions; support repair/selection replayed two plus one. Matched
routing replayed 86 Qwen hits; projection replayed 47. Both selection SHAs
remained identical. SDK retries were zero throughout.

This continuation completed 42 Terra and 186 Qwen gateway calls: **228 completed
calls**, including the failed compilation policies' saved completions. The
earlier r2 root separately retains eight raw request reservations and zero
response journals after local socket failures. These are not counted as
successful completions or cleared to enable retries. Its abort receipt records
the environment issue: the normal process could read the original DB but could
not open a socket; the network-enabled process could not read that DB. The r3
successor stages only the approved turns in a worktree SQLite file and verifies
the unchanged population SHA
`8b5d302f977ad5143aa20ba32a8cbf9b60134fce63c3c7d9b0fb8267dcf91f73`.

Focused integration validation: **115 passed in 8.27 seconds**. It covers user
ownership, summary-only Qwen input, attention boundaries, bounded exploration,
source scope, exact hydration, malformed outputs, summary overflow, singleton
reuse and user-channel projection. Three added lifecycle tests use synthetic
inputs and an in-process fake completion client only. They exercise answer
deduplication and logical-row alignment, late reference loading, separate group
denominators, rejection of malformed judgments, and identical answer/judge
replay without creating a client. These tests do not generate real predictions
or resolve the answer-purpose approval block. Existing unrelated worktree
changes remain.

## Prepared answer evaluation and approval boundary

The user said "Approve sending raw" and then requested a real-data evaluation.
Raw compilation completed. Automatic approval review subsequently rejected
Terra **answer generation**, stating that it considered the approval limited
to summary compilation. It upheld that rejection after a local audit proved
every packet was an exact subset of the same approved turns. No answer request
process launched and no answer checkpoints exist. This is an automatic-review
block, not a skill requirement or a request to approve compilation again.

The reviewable remaining action is sealed in
`user-spine-real-matched-pilot-20260909-r1/answer-judge-approval-request.json`, SHA
`714b84d843a4760c18845b680e9e8a0123329b30d7081383e3766fdc11345fc1`:

- Forty logical answer packets across five arms, containing exact sections from
  30 of the same 39 turns. No new raw source content.
- At most 37 unique Terra answer calls across the two frozen phases (22 + 15),
  maximum 256 output tokens each.
- Then at most 40 Sol judge calls, maximum 32 output tokens, carrying the
  question, reference answer and generated answer rather than whole transcripts.
- Same local gateway, `https://central-dev.zt:4000/v1`; zero SDK retries; no raw
  content to Qwen. Generated answers remain unknown until this action runs.

The seven development support references were joined for the separate local
coverage audit **after routing and answer prompts sealed**, because provider
answering was blocked. The benchmark gold has not been joined in this run.
Subsequent answer generation must reuse the sealed prompts without incorporating
reference-derived changes. Judge preflight loads benchmark gold only after
answer artifacts seal and reports the two question populations separately.

After explicit approval for these purposes, continue from the worktree with
the network-enabled command environment and the already authorized gateway:

```powershell
$spinePython = '.\.pixi\envs\dev\python.exe'
$spineMatched = 'eval_results/user-spine-real-matched-pilot-20260909-r1'
$spineProjection = 'eval_results/user-spine-real-projection-pilot-20260909-r1'
& $spinePython -X utf8 -m tools.evaluate_user_spine_real_pilot answers --output-root $spineMatched --enable-provider
& $spinePython -X utf8 -m tools.evaluate_user_spine_projection answers --output-root $spineProjection --enable-provider
& $spinePython -X utf8 -m tools.evaluate_user_spine_real_pilot judge-preflight --output-root $spineMatched
& $spinePython -X utf8 -m tools.evaluate_user_spine_projection judge-preflight --output-root $spineProjection
& $spinePython -X utf8 -m tools.evaluate_user_spine_real_pilot judge --output-root $spineMatched --enable-provider
& $spinePython -X utf8 -m tools.evaluate_user_spine_projection judge --output-root $spineProjection --enable-provider
```

Run reference-loading preflight in the normal environment if the network-enabled
worker cannot read the dataset. Replay answer and judge phases without
`--enable-provider`, authenticate zero new calls, inspect chronology errors, and
append measured answer results. Do not relaunch compilation or reseal routing
under changed prompts. Larger independent multi-question-family validation
remains necessary before a routing or answer-quality promotion.
