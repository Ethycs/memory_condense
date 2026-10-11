# BGE-only tactic prioritization

**Status:** Automatic proxy dispatch and combined neighborhood tactic included in v0.2.2-beta.1

**Date:** 2026-10-08

**Applies to:** focused recall versus dialogue-wide retrieval

**Depends on:** [Dialogue-wide retrieval experiments](263%20-%202026-10-06%20-%20Dialogue-wide%20neighborhood%20retrieval.md)

The existing BGE-M3 FP32 FastEmbed CPU encoder can classify which retrieval
tactic to prioritize. A three-class linear head adds 63,719 bytes to the package
and needs no second transformer, generative completion, Qwen attention pass,
GPU allocation, or provider call. This makes the classifier a one-local-model
component; it does not remove the other models used elsewhere in the system.

The outputs are `focused` (prioritize `cap8`), `overview` (prioritize
`hierarchy_neighborhoods`), and `uncertain` (abstain). Recent messages or summaries
can disambiguate short follow-ups. The classifier never reads the archive.
The implementation and persistent CLI are described in the
[proxy guide](../02%20-%20Implementation/06%20-%20Provider%20Proxy%20and%20Transcript%20Import.md#3-local-tactic-prioritization-2026-10-08).

## Confirmation results

| Partition | Linear head | Nearest-centroid baseline |
| --- | ---: | ---: |
| Fresh authored scope/context cases | **44/46 (95.7%)** | 42/46 (91.3%) |
| New factual questions from two existing history sets | **199/200 (99.5%)** | 200/200 (100%) |
| Existing dialogue-wide development questions | **20/20 (100%)** | 20/20 (100%) |

The authored set contains 20 focused, 20 overview, and six underspecified cases.
The head got 19/20 focused, 19/20 overview, and all six underspecified cases
right. Three of four paired follow-ups were right in both contexts (7/8
individual decisions). The factual set had one false overview selection.
These are scope labels, not measured answer accuracy or proof that the selected
tactic produces the best answer on each question. Do not pool the partitions
into a headline percentage: the 200 factual negatives dominate that aggregate.

All five live JSONL smoke requests also behaved as expected: a port lookup
selected focused recall, a project retrospective selected overview, and
"Is this working?" abstained alone, selected focused with a single-function
context, and selected overview with a project-evaluation context.

## First-pass failure and adaptation

The first frozen run trained on 88 authored examples and selected parameters
on 40 development examples. It scored 60/62 on its authored test but only
76/100 on real factual queries: 24 were incorrectly sent to overview. Requests
for a particular past opinion, topics, challenges, or plans often contain
language resembling synthesis even though they ask for concrete facts.

Those exposed 100 questions then became development data, with a fixed
70-training/30-development split. They are not fresh validation for the revised
head. The second run used 158 training and 70 development examples, 46 newly
authored confirmation cases, and 200 different factual questions from the next
two history IDs in the saved 5M battery. No history was ingested again and no
answer model was called. The original first-pass report is retained unchanged.

All second-stage partitions were frozen before embedding/fitting. The fitting
phase reads only training/development labels. It selected regularization C=10
from the recorded grid, with 69/70 correct on development. No additional score
or margin cutoff was selected; abstention comes from the learned `uncertain`
class. Softmax scores are not calibrated probabilities. No confirmation-driven
retuning followed the reported errors.

## Remaining errors and evidence limits

The three confirmation errors were:

1. "Which repeated requests from me have remained insufficiently addressed?"
   selected focused instead of overview.
2. "Have we made progress?" with context about one upload moving from 40% to
   65% selected overview instead of focused.
3. A factual question asking what one graphic-designer collaboration achieved
   and what follow-up was planned selected overview instead of focused.

The 20 historical overview questions are development sanity checks, not an
independent test. New factual questions come from the same question-generation
pipeline as the first 100, and their focused labels reflect the intended
factual-QA default, not a per-question tactic comparison. Authored examples share
one author and taxonomy. Context tests use short authored summaries; a real
rolling-chat evaluation remains necessary before enabling automatic dispatch.

## Cost and integration boundary

Across 494 second-stage inputs, actual CPU embedding time averaged **0.131 s**,
with median **0.117 s** and p95 **0.215 s**. Applying the head to cached embeddings
averaged **0.093 ms**. Second-run cold loading took **6.73 s**; the first cold
load took **21.34 s**. These startup costs reinforce keeping BGE resident.
A compatible, already-computed embedding of the exact classifier input can
eliminate the extra encoder call.

At the classifier-only checkpoint, `BGETacticRouter`, `LocalRuntime.tactic_router()`, and
`memory-condense route --jsonl` were implemented. The packaged head matches the
evaluated artifact. At that checkpoint the proxy's defaults were unchanged. The combined
hierarchy/neighborhood packet assembly was still an evaluation path; predicting
its name did not enable it in the proxy. No release was built or published for
that classifier-only change. The subsequent integration below supersedes this
dispatch limitation.

The affected suite passed **48 tests**, including exact embedding reuse,
encoder drift/corruption rejection, bounded context, one persistent CLI encoder,
packaged-head loading without training dependencies, and runtime encoder sharing.
`git diff --check` passed. The new classifier tests are included in CI.

## Artifacts

- [Original first-pass report](../../eval_results/bge-tactic-router-20261007-r1/report.json)
- [Confirmation report and every decision](../../eval_results/bge-tactic-router-20261007-r2/report.json)
- [Frozen confirmation plan](../../eval_results/bge-tactic-router-20261007-r2/plan.json)
- [Training/development selection](../../eval_results/bge-tactic-router-20261007-r2/fit.json)
- [Live CPU CLI smoke outputs](../../eval_results/bge-tactic-router-20261007-r2/live-cli.jsonl)
- [Final artifact audit](../../eval_results/bge-tactic-router-20261007-r2/final-audit.json)
- [Affected test results](../../eval_results/tactic-router-tests.xml)

The classifier file SHA-256 is
`ce170fff6e2ff56d1fd8fd85f345325c4c41b22155160c3b97071ab57e2bf9c4`.
CLI decisions identify its canonical payload receipt as
`88fd78b03dfa7b9ca7248bdb9cd56c950a85981f0f0f503dfee95502dea33146`;
these are deliberately different hashes (whole file versus payload without
its receipt field).

## Enabled proxy integration (2026-10-08)

The proxy now defaults to `--memory-retrieval-policy auto`. It shares the runtime's
existing CPU BGE encoder and packaged head. A clear question reuses the retrieval
query vector without an additional encoder pass. An uncertain decision can use
one more embedding with the preceding six user/assistant messages, bounded to
256 context tokens. If still uncertain, or if the question exceeds the classifier's
512-token limit, retrieval uses cap-8. Encoder identity drift still fails validation.
Explicit `cap8`, `dialogue`, and `hierarchy_neighborhoods` modes bypass classification.

Overview decisions now produce a combined packet: authenticated hierarchy
summaries plus the existing exact closing-response neighborhood evidence. The
summary renderer binds chronology to the published transcript's turn positions,
not content-hash ordering. It includes the whole eligible hierarchy when it fits
within 24,576 tokens. Otherwise selected neighborhoods and their ancestors get
priority, summaries are omitted whole, and coverage counts identify the omissions.
The existing raw profile defaults remain 6,144 hydration tokens and 128 spans.
Envelopes, recent exchanges, provider instructions and replies add tokens beyond
these two context budgets.

The summary block identifies summaries as historical reports, with H labels and
original source/turn pointers. Exact raw passages retain the existing MEMORY
envelopes and verified span receipts. Only those delivered raw references feed
the existing Hebbian learning path; summary pointers cannot manufacture raw
evidence or independent learning credit. Recall journal events retain the
classifier decision, chosen tactic, summary coverage/pointers, and actual packet
size. Retries keep their original packet and response.

The real-model integration replay reused the same 775,741-token engineering
memory without re-ingestion or new summary generation. All 24 routing cases
selected the expected tactic: 20 existing retrospectives, one factual question,
one underspecified question, and two context-dependent follow-ups. Every one of
the 20 retrospective packets preserved the exact ordered neighborhood spans
from the earlier assay. All 51 eligible hierarchy summaries fit, with original
summary content and explicit transcript ordering. This is dispatch and evidence
parity, not a new answer-quality score or independent classifier validation.

The first replay exposed index-ID ordering in the summary renderer during final
review. It is retained as r1; the corrected chronology-bound replay is r2. Source
memory files were unchanged in both runs. No provider calls or new answers were
used. The current renderer and dispatcher are checked by the r2 manifest.

The corrected replay assembled broad packets in **1.10 seconds mean** and
**1.35 seconds maximum**, excluding the **5.64-second** model startup. This
includes classification, routing, hydration, and packet construction, but no
answer generation. All **135 affected tests passed**, covering automatic and
forced modes, contextual fallback, source/date scope, summary budgets and
chronology, failed-publication isolation, exact-source learning, retry replay,
and both provider HTTP formats. `git diff --check` and CLI help passed.

The implementation is enabled in the checkout and the next package build.
The already-published 0.2.1 artifact was not replaced and no running service was
restarted as part of this change.

- [Corrected real BGE integration report](../../eval_results/automatic-tactics-20261008-r2/report.json)
- [Replay plan and source bindings](../../eval_results/automatic-tactics-20261008-r2/plan.json)
- [Complete affected test results](../../eval_results/auto-route-complete.xml)
- [Integration audit](../../eval_results/automatic-tactics-20261008-r2/final-audit.json)

## Query-driven decision and fix history (2026-10-08)

The user's selected extension is query-time recall of decisions, attempted
fixes and their evidence/revisions. It does not introduce a continuously
extracted project-state ledger or cross-conversation namespace federation.

A separate binary linear head shares the same existing BGE query vector. It
selects `decision_history` before falling back to the unchanged three-class
scope result. Only an uncertain overall decision can request the existing
bounded recent-context embedding. The additional sealed head is 21,659 bytes.
The first fit selected a conservative threshold on 82 training and 30
development examples. It got 32/40 confirmation cases correct: no false
positives, but eight missed history requests (10/18 positive recall). That
report remains unchanged. The 40 exposed cases became development data, and
36 new training examples broadened outcomes/correction phrasing. The second
fit used 118 training and 70 development cases, choosing C=100 and threshold
0.7 on development balanced accuracy with a false-positive penalty.

The new, frozen 52-case confirmation returned **51/52**: **25/26** positives and
**26/26** negatives. The miss was the contextual follow-up “Did it help?”. None
of 200 cached factual vectors from the previous confirmation triggered the new
head. Those 200 are a dispatch regression check, not fresh hand-labeled history
intent ground truth. Authored partitions share an author and taxonomy. Mean
new-vector embedding time was 0.104 seconds; production normally reuses the
retrieval query vector, so this is not an added encoder cost.

The retrieval path uses the existing hierarchy plus neighborhood packet. It
adds bounded chronological annotations, speaker attribution, exchange links
and D labels on exact passages. Existing canonical memory revisions are frozen
at publication and exposed only with both independent support quotes delivered.
Chronology is never labeled supersession. The reader is asked to separate
proposals, attempts, reported implementation, tests and releases, and to retain
uncertainty. Native summary archives may have zero stored typed revisions.
Reader-inferred corrections remain answer interpretations, not new stored facts.

A read-only replay used the same 775,741-token, 481-message engineering memory
and three existing questions: Q04 (reversed decisions), Q06 (bottleneck fixes),
and Q12 (unresolved problems). Every original raw reference and hydration
receipt matched the control. All 19, 8 and 7 delivered spans were annotated.
Packets grew by 1,148, 580 and 530 tokens respectively. History packet assembly
took 1.09–1.22 seconds after model startup. No history was re-ingested and no
summaries were rebuilt.

Six new reader requests compared the same Sol model, reasoning disabled,
against the same evidence with and without annotations. Selective checkpoint
coverage tied on Q04 (75%) and Q06 (50%), and increased from 50% to 60% on Q12.
The mean was 58.3% versus 61.7%. These are three development questions and a
selective, model-graded rubric: not general QA accuracy, not a statistically
established improvement, and not comparable to the factual 100-question scores.
The judge's additional unsupported-claim flags only consulted its selective
reference quotes; they are not valid hallucination counts without checking the
full delivered evidence. For example, some flagged commit/test reports were
present in the retrieved passages and explicitly attributed as reports.

Artifacts:

- [First confirmation, retained](../../eval_results/history-intent-20261008-r1/report.json)
- [Fresh second confirmation](../../eval_results/history-intent-20261008-r2/report.json)
- [Matched packets and source parity](../../eval_results/decision-history-engineering-20261008-r1/packets-report.json)
- [Matched reader checkpoint results](../../eval_results/decision-history-engineering-20261008-r1/answer-report.json)

All **343 configured release tests passed locally**, including both provider
protocols, forced and automatic routing, failed-publication isolation, original
source learning, revision endpoint validation and annotation budgets. Results:
[release test XML](../../eval_results/decision-history-release-tests.xml).

The checkout enables this tactic under automatic routing and provides an
explicit `decision_history` mode. The previously built release bundle predates
this extension and must be rebuilt before publishing it.
