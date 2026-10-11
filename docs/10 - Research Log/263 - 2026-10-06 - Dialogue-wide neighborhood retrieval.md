# Dialogue-wide neighborhood retrieval

**Status:** experimental policy integrated as an opt-in packaged runtime module;
production cap-8 defaults unchanged. Tuning and release publication are pending.
**Date:** 2026-10-06

The user asked whether existing neighborhoods could answer questions whose
conclusions were never explicitly stated: whether the system works, what mistakes
were made, and what approaches were tried. This experiment compares retrospective
answers over one existing engineering memory, with exact original-source evidence.

## Scope and controls

The source is the complete persisted memory in
`eval_results/native-spine-engineering-live-session-20260922-r1/memory`:
**481 messages, 775,741 source tokens, 783 atomic summaries, 26 attention-selected
leaf neighborhoods, and 51 hierarchy sections**. The prior session also had
43 later journal messages that were never successfully indexed. They are excluded
from every arm and the reference. The archived initial seed omitted its original
tool dumps; newly generated tool observations were stored. This evaluates the
stored memory as it existed, not the current repository's implementation status.

Fifteen questions cover goals, attempted approaches, mistakes, reversals,
architectural evolution, latency versus other costs, accuracy changes, benchmark
comparability, supposedly fixed problems, delivery states, retained mechanisms,
outstanding work, measured benefits, recurring failure patterns, and beta readiness.
The source stays read-only, and generated answers do not enter subsequent recall.
There is no extra recent-context tail. This isolates retrieval for retrospective
questions rather than exercising continuous proxy ingestion or its recent window.

The initial arms are production cap-8 with its **2,048-token** hydration limit,
the same cap-8 route with **6,144 tokens**, and a neighborhood survey with the
same **6,144-token** limit. All use the same Sol model, disabled reasoning,
system instructions, questions, output limit, and evidence formatting. Actual
sent token counts can differ because exact sections must fit intact. No whole
transcript is sent to the API.

## Mechanism and observed repair

`DialogueNeighborhoodRouter` embeds the 26 stored leaf summaries with the
admitted FP32 CPU FastEmbed BGE model. It selects up to twelve neighborhoods,
reserving chronological coverage across four regions, then chooses semantic
seeds with different speaker roles and nearby turns. All addresses remain bound
to original source spans. Routing reads summaries and topology only; the existing
exact hydrator reads raw content afterward. It preserves source-date eligibility,
rejects changed source text, and never certifies exhaustive factual coverage.
The initial setup reused all historical atomic vectors and attention topology;
summary/query embedding and cold encoder setup took **56.461 seconds**. There
was no history re-ingestion, Qwen inference, or local summary generation.

The first variant admitted more assistant evidence, but often selected intermediate
activity inside the correct exchange while missing its closing response. For
example, all fifteen initial neighborhood packets omitted T450, the later
237-test delivery statement. Answers could therefore treat earlier migration
failures as unresolved.

One development follow-up preserves the same selected neighborhoods, questions,
reference, vectors and 6K limit, but prioritizes each neighborhood's closing
assistant turn before its other candidates. A leaf ending in a tool/system or
user turn receives no synthetic closing answer. This is a metadata-based
reservation, not a guarantee that the assistant's final claim is true. It brought
T450 into **11/15** packets and the first reviewed answer correctly distinguished
the early failure, subsequent reported fix, and remaining commit/benchmark work.
No new embeddings or history builds were needed for this adjustment.

The follow-up is development on the same questions after inspecting initial
misses. It is not a held-out confirmation. There is still no automatic
reader-directed follow-up query or Hebbian graph traversal in this experiment.

## Scoring and validation

Before answer generation, 24 reference facts were authored from original messages
and their supporting quotations verified against exact source offsets. The
fifteen questions require **70 question-specific checkpoints**, graded 0 for
absent/incorrect, 1 for partial, or 2 for substantive coverage: **140 possible
points** for a complete arm. Historical assistant claims are identified as claims.
The reference is selective, not an exhaustive annotation of 776K tokens.

Sol grades paired answers with route names hidden and label order permuted by
question. The generator and grader are the same model, so this is not independent
adjudication. Citation labels are also checked against the evidence actually
provided. These rubric scores are not comparable to the earlier factual-recall
accuracy percentages. Q01's shared “working against our stated goals” wording
was ambiguous; the rubric scores evidence coverage rather than a binary yes/no.

All **45 initial answers** completed. Two initial grading calls timed out; their
reservations and the original answer outputs remain preserved without automatic
resend. The thirteen successfully graded initial triplets scored **27/120** for
cap-8/2K, **43/120** for cap-8/6K and **56/120** for the first neighborhood policy.
These are incomplete pilot results, not fifteen-question aggregate scores.

The closing-response follow-up adds fifteen answers and a separate blinded
comparison against the existing two 6K arms. All **60 answers** and all **15 final
grading triplets** completed. The complete comparison uses one grading round;
its baseline scores are not mixed with the incomplete initial pilot.

| Retrieval policy | Checkpoint points | Coverage score | Mean evidence tokens | Mean API answer time |
| --- | ---: | ---: | ---: | ---: |
| Standard cap-8, 6K limit | 50/140 | 35.7% | 4,909 | 28.54 s |
| Neighborhood survey, 6K limit | 57/140 | 40.7% | 5,729 | 32.70 s |
| Survey with closing responses, 6K limit | 83/140 | 59.3% | 5,794 | 30.24 s |

The closing-response policy beat the standard route on **12 questions**, tied
one and lost two: **+23.6 percentage points** in checkpoint coverage. Against
the first neighborhood policy, it won ten, tied one and lost four. Its realized
evidence length was 18.0% higher than the standard route, despite an identical
maximum budget; this is not an exact-token-matched ablation. Relative to the first
neighborhood policy, it used only 1.1% more evidence tokens.

These times are observed API-call wall times under three concurrent requests,
not production chat latency or end-to-end ingest timings. Mean exact hydration
was 0.073 s, 0.689 s and 0.878 s respectively. Query embeddings were cached;
routing measurements therefore exclude live query embedding and cold setup.

### What improved, and what still fails

- **Architecture evolution (Q05):** standard routing scored 4/10 and the
  closing-response policy 9/10. It connected earlier bypassed evaluation and
  ineffective decay to later delivered changes.
- **Delivery state (Q10):** 2/10 became 7/10. The revised answer separates
  proposals, tests, measurements and unfinished delivery, but still misses the
  earlier merge reported at T79; that turn was absent from its packet.
- **Latest state (Q01):** 3/10 became 7/10. The revised answer now reconciles the
  earlier evaluation leak and stale documentation with T450's reported fixes,
  237 passing tests and remaining commit/benchmark work. It still overstates the
  goal as requiring better recall *and* lower cost; T169 allowed token savings
  as an alternative. T169 was absent from this packet.
- **Regressions remain:** mistakes (Q03) fell from 4/8 to 3/8, and unresolved
  work (Q12) from 5/10 to 4/10 versus standard routing. Versus the first survey,
  closing-response priority also lost useful accuracy and cost evidence on
  Q07 and Q13. A closing reply can displace details elsewhere in the exchange.

All explicit bracketed citation labels refer to messages present in their
respective packets. This validates source availability, not claim entailment.
The original citation counter falsely counted prose references to the archive
cutoff T480; the corrected deterministic report preserves model scores and
original receipts. The judge flagged reference contradictions in 6, 7 and 2
answers respectively, but some flags penalize honest statements about missing
packet evidence. For example, Q13's standard answer correctly says its supplied
evidence lacks the later control result. Treat those counts as grader flags,
not independently verified error rates. A separate source review records this
qualification without changing checkpoint scores.

**Conclusion:** the existing neighborhoods can support better dialogue-wide
reasoning when retrieval reserves outcomes as well as intermediate activity.
This development test supports that mechanism; it does not establish complete
retrospective recall. The remaining issue is evidence coverage and temporal
reconciliation. The next useful experiment is a bounded second recall for
unresolved or contradictory claims, followed by held-out questions or another
existing memory. The experimental route remains outside production defaults.

The frozen Q01 wording remains unchanged in every completed answer. Future
preparations use the unambiguous wording “Does the system meet our stated goals?”

**Regression checks:** 20 tests passed, covering the new policy, existing parent
context and user completion, dated-source isolation, exact hydration, rejection
of corrupted raw sources, encoder identity, closing-response ordering and an
open neighborhood with no completed assistant response, plus citation parsing.
The new tests are added to CI. This is not a live proxy or GPU throughput benchmark.

## Evidence

- [Frozen source snapshot](../../eval_results/dialogue-neighborhood-20261006-r1/source.json)
- [Fifteen questions](../../eval_results/dialogue-neighborhood-20261006-r1/questions.json)
- [Source-verified private checkpoints](../../eval_results/dialogue-neighborhood-20261006-r1/private-reference.json)
- [Initial preparation and implementation hashes](../../eval_results/dialogue-neighborhood-20261006-r1/prepared.json)
- [Closing-response development plan](../../eval_results/dialogue-neighborhood-20261006-r1/closing-plan.json)
- [Final blinded scoring plan](../../eval_results/dialogue-neighborhood-20261006-r1/closing-judge-plan.json)
- [Complete comparison with corrected citation validation](../../eval_results/dialogue-neighborhood-20261006-r1/verified-closing-report.json)
- [Source and execution audit, including manual flag review](../../eval_results/dialogue-neighborhood-20261006-r1/final-audit.json)
- [Regression tests](../../eval_results/dialogue-neighborhood-tests-20261006-r3.xml)

The initial router, runner and grader implementations are preserved beside their
frozen receipts as `initial-dialogue-neighborhood.py`, `initial-runner.py` and
`initial-grader.py`. `closing-grader.py` preserves the final grading implementation
before the deterministic citation-counter correction. No model score was revised
to improve the result.

## Requested full-context control: gateway failure

The subsequent requested control preserves the same 481 messages, original
fifteen questions, Sol alias, disabled reasoning, frozen checkpoints and source
labels. Every raw message is rendered in chronological order, including tool
output. The formatted archive is **781,268 local proxy tokens**. Existing memory
answers are reused; no memory is re-ingested and no new memory answers are generated.

After the user clarified that the archive should be sent once, the control was
prepared as **one request containing the full archive and all fifteen questions**:
**781,976 local proxy input tokens**, 15,000 shared output tokens, and a maximum
of 450 words per answer. Each answer would be graded separately. Joint questions
and a shared output budget differ from the independent memory calls, so this is
a batched full-context comparison, not an exactly matched generation protocol.
It does not send summaries or silently truncate the archive.

| Request | Result | Observed duration |
| --- | --- | ---: |
| Preliminary single-question streaming probe | HTTP 200, no answer text or token usage | 20.73 s |
| All fifteen questions, full context once, nonstreaming | HTTP 500, upstream internal server error | 10.19 s |
| Tiny request, same Sol alias and nonstreaming transport | HTTP 200, `OK` | 4.43 s |

The batch gateway error reports `OpenAIException - Internal Server Error` with
no configured model-group fallback. LiteLLM call ID:
`14e89ef2-3674-42f1-8706-e05cfae1ed4e`.
The gateway lists the alias but returns 403 for model metadata. Sol's
[published model specification](https://developers.openai.com/api/docs/models/gpt-5.6-sol)
lists a 1,050,000-token context window, but that does not establish this gateway's
configured limit. The successful tiny request narrows the problem to something
about handling the large request; it does not identify the upstream cause.

**These gateway attempts produced zero answers and no quality score.** This gateway
failure supplies no quality advantage to memory. The exact request and transport
receipts are preserved, with no automatic resend. An upstream error log or a Sol
route accepting the unchanged batch is needed to finish the comparison. The
same class of large-request failure was recorded in Research Log 242 on September
22; that earlier incident also did not establish a context limit.

The control renderer and batch-population checks passed (three tests). The
executed batch runner is archived before a later provenance-only fix separating
batch answers from the failed single-question probe's artifact directory.

- [Control status and request identity](../../eval_results/dialogue-full-context-20261006-r1/comparison-status.json)
- [Single-archive batch plan](../../eval_results/dialogue-full-context-20261006-r1/batch/plan.json)
- [Exact batch request](../../eval_results/dialogue-full-context-20261006-r1/batch/request.json)
- [Gateway response](../../eval_results/dialogue-full-context-20261006-r1/batch/transport.json)
- [Control validation tests](../../eval_results/dialogue-full-context-tests-20261006-r2.xml)

### In-session Astra control: context compaction

At the user's request, an isolated `gpt-6-astra` sub-agent was launched to read
the same complete archive once and answer all fifteen questions without the
gateway. It received no prior answers, packets, scores, or private checkpoints.
The planned control used medium reasoning; the existing memory answers used Sol
with reasoning disabled, so even a completed result would have a model and
reasoning-setting confound.

The source contains **2,703,700 formatted raw characters**. Large tool outputs
were truncated even when larger output budgets were requested, so the agent
recovered the affected ranges using smaller contiguous reads. It successfully
loaded **636,000 characters across 22 complete chunks**, after which the sub-agent
runtime actually compacted its context. Earlier raw text was no longer intact.
The agent stopped rather than substituting the compaction summary for full context.

**Result: zero generated answers and no full-context score.** This establishes a
limit encountered in this sub-agent runtime, not Astra's advertised model window
or its answer quality. No additional gateway calls or memory ingestions were made.
The parent verified the source hash and contiguous chunk offsets. This attempt
required a different runtime to retain the complete raw archive; the local
app-server experiment below subsequently succeeded. Any smaller or filtered-context
test must be labeled separately.

- [Astra control plan](../../eval_results/dialogue-astra-full-context-20261006-r1/parent-plan.json)
- [Verified no-score status](../../eval_results/dialogue-astra-full-context-20261006-r1/parent-status.json)
- [Raw-read and compaction receipt](../../eval_results/dialogue-astra-full-context-20261006-r1/read-receipt.json)
- [Reader outcome](../../eval_results/dialogue-astra-full-context-20261006-r1/reader-result.json)

### Local Astra with an enlarged context: complete raw control

The user then asked whether compaction could be prevented. The in-chat sub-agent
tool exposes no context-threshold override. The installed local Codex CLI
(`0.154.0-alpha.6.2`) does: the cached Astra catalog advertised a **272,000-token
default**, **872,000-token maximum**, and **95% effective context window**. This
is local runtime metadata, not a general statement about all Astra deployments.

An isolated local run used the documented per-run settings:

```toml
model_context_window = 872000
model_auto_compact_token_limit = 820000
```

The settings raise the threshold for this bounded experiment; they do not disable
all future compaction or expand the model beyond its supported capacity. Global
user configuration was not edited. The first CLI attempt hit a separate
1,048,576-character user-input limit. A subsequent file-loader attempt was rejected
by the nested runner's policy and saw no archive. Its fifteen structured inability
responses were explicitly excluded from scoring.

The successful path used Codex's documented `thread/inject_items` operation to
append the **entire original archive as one user message** to a fresh, read-only
Astra thread, then asked the fifteen frozen questions in one turn. No shell loader,
retrieval, intermediate summary, or additional model tool call was involved.
The app-server acknowledged the exact injected artifact. The effective turn
settings confirm **Astra, medium reasoning**, overriding an inherited thread
default of xhigh.

| Completed raw-context run | Observed result |
| --- | ---: |
| Answers | 15/15 |
| Input tokens reported by runtime | 795,136 |
| Output tokens | 5,671 |
| Reasoning output tokens, included in output | 54 |
| Effective model context window | 828,400 |
| Total tokens | 800,807 |
| Context-compaction events | 0 |
| Model tool calls | 0 |
| Model generation requests | 1 |
| Elapsed time through generation, including thread setup | 172.72 s |

The parent independently checked exact injected text, unchanged original source
hashes, reference isolation, complete answer IDs, source citation availability,
runtime input usage, model identity, and absence of compaction/rerouting/tool calls.
All answers are between 205 and 258 words, within the shared 450-word limit.
This admits the result as a complete raw-context control for this archived scope.

**Comparability limits remain:** this control uses Astra with medium reasoning;
the saved memory answers use Sol with reasoning disabled. All fifteen full-context
questions were answered jointly, while memory answers were generated separately.
The comparison measures those configurations and cannot isolate retrieval loss or
model quality. Grading uses the original private checkpoints, concealed route
labels and the same Sol judge. No memory answers are regenerated.

The focused regression suite passed **22 tests**. All fifteen blinded grading
triplets completed, with no grading transport errors in this comparison.

| Configuration | Checkpoint points | Coverage score | Total reader input for 15 questions |
| --- | ---: | ---: | ---: |
| Standard cap-8 memory, Sol / none | 52/140 | 37.1% | 76,203 tokens |
| Neighborhoods with closing responses, Sol / none | 81/140 | 57.9% | 89,483 tokens |
| Full raw archive once, Astra / medium | 88/140 | 62.9% | 795,136 tokens |

Neighborhood memory finished **5.0 percentage points below** the full-context
control on this coverage rubric. It won three questions, tied two and lost ten.
Compared with standard routing, it won twelve, tied two and lost one. These are
coverage checkpoints, not factual-QA accuracy or proof of equivalent engineering
quality. The same stored memory answers previously scored 83/140 in a different
blinded triplet comparison; 81/140 here reflects a new grading round, not a
changed memory answer. Compare configurations within this completed round.

The batched full-context reader actually received the archive once. Consequently,
the neighborhood method used **88.7% fewer total input tokens** for this fifteen-
question battery, not a calculation that pretends the archive was sent fifteen
times. The counts include each runtime's prompt framing and may reflect model
tokenizer differences. Shared full-context latency is not an independent per-
question latency measurement.

Both full context and neighborhood memory scored 9/10 on architecture evolution
and 7/10 on delivery states. Full context recovered the earlier merge claim at
T79 that the memory packet omitted. Full context led 8/10 versus 4/10 on recurring
failure patterns; memory led 8/10 versus 4/10 on retained/bypassed mechanisms.
These differences include evidence selection and answer emphasis, as well as the
different models and reasoning settings.

The full-context answer received one automated contradiction flag, on Q08.
Source review rejected that flag: Astra correctly distinguished the four-sample
pooled recall gain from the conv-26 matched-cost illustration, just as T206 does.
The grader relied on a compressed T211 retrospective description. Automated
checkpoint scores remain unchanged; this separate review prevents that flag from
being reported as a verified reader error.

**Outcome:** compaction was prevented for this bounded complete-context run, and
the requested control is now available. Neighborhood memory approached the larger
control's checkpoint coverage with substantially less input. A same-model,
same-reasoning, held-out comparison is needed to isolate retrieval effects.

- [Successful per-run configuration and protocol](../../eval_results/dialogue-astra-appserver-20261006-r1/plan.json)
- [Independent full-context admissibility audit](../../eval_results/dialogue-astra-appserver-20261006-r1/admissibility.json)
- [Raw reader answers](../../eval_results/dialogue-astra-appserver-20261006-r1/reader-result.json)
- [Generation and token-usage receipt](../../eval_results/dialogue-astra-appserver-20261006-r1/generation.json)
- [Comparison configuration and confounds](../../eval_results/dialogue-astra-comparison-20261006-r1/comparison-context.json)
- [Complete blinded quality comparison](../../eval_results/dialogue-astra-comparison-20261006-r1/full-report.json)
- [Final source, scoring and token audit](../../eval_results/dialogue-astra-comparison-20261006-r1/final-audit.json)
- [Regression checks](../../eval_results/dialogue-control-tests-20261006-r1.xml)

## Packaged runtime module (2026-10-06)

The closing-response policy now has a versioned, immutable
`DialogueRetrievalPolicy` and a prepared runtime module in
`memory_condense.application.dialogue_retrieval`. The native resident backend,
session factory, and installed proxy CLI support it. Enable it with
`--memory-retrieval-policy dialogue`; optional `--memory-dialogue-config` loads
a JSON tuning profile at startup. The
[proxy guide](../02%20-%20Implementation/06%20-%20Provider%20Proxy%20and%20Transcript%20Import.md#dialogue-wide-retrieval-module)
documents parameters, defaults, Python entry points, diagnostics, and rollback.
This enables the policy for every recall on that proxy instance; automatic intent
selection and corrective second recalls remain future tuning work.

Publication prepares the leaf-summary index using the resident BGE encoder and
existing vector cache. Unchanged summaries reuse vectors across appends and
cold reopening. A recall embeds its query once, scores summaries, then uses the
existing exact raw hydrator. No new summarization, query-time Qwen pass, or model
loading was added to recall. Failed preparation retains the previous published
view. Original recall ancestry and Hebbian learning continue through the normal
IO path. Compact policy hashes, selected neighborhood IDs, and packing counts
are retained on the recall journal event; these internal diagnostics are not
added to provider messages.

Validation:

- **297 tests passed** in the local run of all 27 CI test files, including the
  new module/lifecycle tests and CLI/package boundaries. One existing Starlette
  deprecation warning remains. Earlier focused checks also passed.
- **15/15 evidence packets matched** the saved closing-response assay: same
  exact spans, order, selected neighborhoods, and hydrated token counts across
  the 775,741-token engineering memory. The replay used authenticated cached
  query/leaf vectors and the shipping native packet entry point. It made zero
  API calls, fresh embeddings, or history re-ingestions; source files stayed
  byte-identical. This is integration parity, not another answer-quality score.
- **Pixi package built successfully**. The extracted package's changed runtime
  files match the checkout, import outside the source tree with research imports
  blocked, and expose both CLI options. This probe uses existing dependency
  installations; hosted CI retains its separate-environment installation gate.
- Default cap-8 behavior remains covered. Both retrieval paths share budget
  resolution, exact provenance validation, and the existing learning path.

The module and CI wiring are ready for tuning and the next release build. No
release was published, version advanced, additional model quality claimed, or
new GPU/provider latency benchmark run in this integration task. JSON-only
policy tuning needs a service restart, not architectural changes or a rebuild.

Artifacts:

- [Full local CI test result](../../eval_results/dialogue-module-ci.xml)
- [Exact-source replay and implementation hashes](../../eval_results/dialogue-module-20261006-r1/report.json)
- [Built-package verification](../../eval_results/dialogue-module-20261006-r1/package-report.json)
- [Local Pixi build log](../../eval_results/dialogue-module-20261006-r1/build.log)

## Pure index dumps and combined evidence (2026-10-06)

The user requested three pure index dumps and a combined arm against the saved
control. The experiment reused the same 775,741-token memory, original 15 question
wordings, and 70 frozen reference checkpoints. Four new Sol/no-reasoning arms
generated 60 answers. The earlier 15 neighborhood-evidence answers and 15
full-context Astra/medium answers were reused unchanged. Fifteen grading calls
scored all six answers together per question under shuffled labels.

The three pure dumps contain every stored summary at their respective level:
26 leaf neighborhoods, all 51 hierarchy nodes, or all 783 atomic summaries.
Compact dumps contain 4,099, 15,359, and 31,988 tokens respectively, before prompt
framing. The entire hierarchy already includes the neighborhood summaries, so
the combined arm adds the earlier closing-response **raw neighborhood packet**
to the hierarchy dump. Nothing was re-ingested, re-summarized, or embedded.

| Arm | Points / 140 | Checkpoint coverage | Total reader input, 15 questions |
|---|---:|---:|---:|
| Saved full raw control, Astra | 87 | 62.1% | 795,136 |
| Saved neighborhood evidence, Sol | 79 | 56.4% | 89,483 |
| Pure neighborhood index | 77 | 55.0% | 65,553 |
| Pure entire-hierarchy index | 78 | 55.7% | 234,453 |
| Pure atomic index | **93** | **66.4%** | 483,888 |
| Entire hierarchy + neighborhood evidence | **91** | **65.0%** | 321,533 |

The prior full-context score was 88/140 (62.9%); the prior neighborhood score
was 81/140 (57.9%). The answers did not change. Their new scores reflect a joint
blinded regrade, not another generation or a system regression. The chart also
shows the original 62.9% control grade as a reference line.

Atomic summaries scored highest in this assay, winning/tieing/losing 8/4/3
questions against the saved full-context answers. The combined arm was 5/6/4
against full context and 11/1/3 against neighborhood evidence alone. It scored
two points below the atomic dump while using **33.6% fewer input tokens**.
Against the full-context battery, atomic and combined input reductions were
39.1% and 59.6%. These totals count the raw control's archive **once**, since
it answered all questions in one shared-context request. Every new arm answered
each question separately. Provider prompt caching discounts are not priced.

Source review identified a useful failure pattern in the coarse dumps. For
example, N0011 retains an earlier pending memory-arm result, and N0013–N0015
retain requests for replication. Answers then reported those states as unresolved
despite the later archived negative cost result (T178) and completed replication
(T206/T211). Atomic summaries preserve more of those later outcomes. However,
the atomic summary labeled A0206 also compresses the pooled 10.3%→23.4% result
and the separate matched-cost comparison into one sentence; its reader repeated
that conflation. Thus finer summaries help coverage but still require exact
evidence to resolve precise numeric comparisons.

**Display-label limitation:** this completed run rendered index turn ranges
using the database's one-based ordinals, while the saved raw passages use
zero-based `T` labels. All **860 summary rows** retain the correct summary IDs,
source IDs, summary hashes, and `N`/`H`/`A` aliases. The combined arm nevertheless
had inconsistent displayed range numbers. Its effect on scores is unmeasured.
The future renderer now uses the control's zero-based convention, with a
regression test. Original requests, answers, and grades are preserved; corrected
index previews were not sent to the reader. The executed runner is archived
beside the results so its frozen implementation hash remains verifiable.

All 60 new reader calls and 15 grading calls completed without failures,
truncation, or retries. Source files and frozen references stayed unchanged.
All explicit citations name supplied entries/passages; this validates source
availability, not factual entailment. Seven focused evaluation tests pass.
The initial r1 preparation stopped before any new API call because a labels
artifact needed an object envelope; the completed assay is r2.

This is selective retrospective coverage on one development episode, not the
earlier factual-recall accuracy benchmark. The Astra/Sol models, reasoning
settings, joint-versus-independent question generation, citation instructions,
and display-label limitation prevent claiming isolated retrieval superiority.
The evidence supports testing an index overview plus exact evidence further;
no production routing default was changed by this experiment.

Artifacts:

- [Comparison chart](../../eval_results/dialogue-index-dump-20261006-r2/charts/comparison.png)
- [Per-question chart](../../eval_results/dialogue-index-dump-20261006-r2/charts/per-question.png)
- [CSV](../../eval_results/dialogue-index-dump-20261006-r2/charts/comparison.csv)
- [Complete report](../../eval_results/dialogue-index-dump-20261006-r2/report.json)
- [Execution and source audit](../../eval_results/dialogue-index-dump-20261006-r2/final-audit.json)
- [Display-label/source-binding audit](../../eval_results/dialogue-index-dump-20261006-r2/label-audit.json)
- [Regression tests](../../eval_results/dialogue-index-label-tests.xml)

Configuration and history-loading behavior follow the official
[configuration reference](https://learn.chatgpt.com/docs/config-file/config-reference)
and [app-server protocol](https://learn.chatgpt.com/docs/app-server).

## Five-question extension across all six approaches (2026-10-06)

The requested extension adds Q16–Q20: shipping recommendations, fixed-token
budget selection, reopening after inactivity, benchmark diagnosis, and integration
failure analysis. These are new scenario questions over the same archived memory
and existing source-verified facts, not five new histories or an independent
held-out corpus. Their mappings to 24 checkpoint occurrences were frozen before
generation: **48 possible points per approach**.

All six approaches generated five new answers. Sol/no-reasoning made 25 separate
memory-reader calls; Astra/medium answered the five control questions together in
one fresh request. The old ephemeral control process had closed, so this request
loaded the complete archive once. It used **794,876 input tokens**, completed in
90.36 seconds, and had no observed compaction or tool calls. No prior answers or
private checkpoints entered any reader prompt. Five blinded grading calls scored
the six answers together per question. The original fifteen grades remain fixed.

| Approach | New points / 48 | New five coverage | All twenty points / 188 | All twenty coverage |
|---|---:|---:|---:|---:|
| Full raw control, Astra | **35** | **72.9%** | 122 | 64.9% |
| Neighborhood raw evidence, Sol | 27 | 56.2% | 106 | 56.4% |
| Neighborhood index only | 21 | 43.8% | 98 | 52.1% |
| Entire hierarchy index only | 20 | 41.7% | 98 | 52.1% |
| Atomic index only | 31 | 64.6% | **124** | **66.0%** |
| Hierarchy + neighborhood raw evidence | 31 | 64.6% | 122 | 64.9% |

On the new five, both atomic and combined evidence finish four checkpoint points
(8.3 percentage points) behind full context, each winning one question, tying
one, and losing three. Across twenty, atomic remains two points ahead of control
and combined evidence ties it. This extension therefore does **not** reproduce
the original small memory lead on its own. It does preserve the practical finding
that atomic summaries and hierarchy plus exact evidence are the strongest tested
memory options; dumping only coarse summaries is substantially weaker here.

The combined arm used **107,159 reader input tokens**, versus **161,418** for
atomic summaries, for the same 31/48 coverage: **33.6% less input**. Neighborhood
evidence used 29,809, the neighborhood index 21,973, and the hierarchy index
78,273. Each memory total sums five independent requests; the control total counts
one shared archive request. Across both rounds, actual input totals are 1,590,012
for control, 645,306 for atomic, and 428,692 for combined. The increased apparent
savings relative to the first round also reflect loading the control archive for
a second batch; they are not evidence of a new compression improvement.

Source review again found the numeric conflation in Q17: atomic, combined, and
neighborhood readers merge four-sample pooled gains with the separate conv-26
matched-budget result. T206 distinguishes them, whereas the later handoff T211
compresses both into one sentence. This failure can therefore propagate through
raw neighborhood evidence as well as stored summaries.

One automated contradiction flag against the full control is a reference-scope
error. The judge said token-target pooling was merely proposed. Later raw code
in T473 shows token-based `span_levels` with defaults `(110, 220)`; the long-form
regression remained unperformed, as the answer states. The source review records
that false flag separately. No checkpoint scores or scope flags were adjusted,
and the review is not an exhaustive adjudication of every answer.

All **860 index rows** in the extension have verified zero-based ranges, source
IDs, and exact summary hashes. All hydrated raw spans match the archive. The
production `PreparedDialogueRetrieval` module prepares the new neighborhood
packets, reusing cached episode vectors and embedding only five new questions.
There were no history re-ingestions, new summaries, API failures, truncations,
or retries. The initial r1 preparation stopped on an encoder-identity check before
any API calls; r2 uses the existing admitted FastEmbed compatibility path.
Twenty-six focused retrieval/control/index tests passed.

The chart separates original fifteen, new five, and weighted twenty-question
coverage. Model/reasoning and joint-versus-independent generation still differ.
The corrected display ranges also distinguish this extension from the original
fifteen. These are selective development-checkpoint scores, not general QA
accuracy or a controlled estimate of retrieval loss. No production default or
release was changed.

- [Updated chart](../../eval_results/dialogue-index-extension-20261006-r2/charts/comparison.png)
- [CSV](../../eval_results/dialogue-index-extension-20261006-r2/charts/comparison.csv)
- [Five-question and aggregate report](../../eval_results/dialogue-index-extension-20261006-r2/report.json)
- [Frozen questions](../../eval_results/dialogue-index-extension-20261006-r2/questions.json)
- [Execution/source audit](../../eval_results/dialogue-index-extension-20261006-r2/final-audit.json)
- [Raw-source review of factual issues](../../eval_results/dialogue-index-extension-20261006-r2/source-review.json)
- [Focused tests](../../eval_results/dialogue-extension-tests.xml)
