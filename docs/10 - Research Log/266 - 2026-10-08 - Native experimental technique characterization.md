# Native experimental technique characterization

**Date:** 2026-10-08  
**Status:** Complete; retain baseline defaults

## Scope and reuse

This follow-up to [265](265%20-%202026-10-08%20-%20Native%20experimental%20module%20integration.md)
tests usefulness, beyond adapter execution. It reuses one already-ingested
775,741-token engineering archive (481 messages, ending at T480), its native
summary hierarchy, BGE vectors, checkpoint-bound CAV artifact and source-bound
reference checkpoints. No history was reingested. This is an archival engineering
reasoning assay, not a replacement for the 1M/5M retrieval benchmarks.

The uploaded-document/file extension is deferred in the existing
[continuity roadmap](../06%20-%20Roadmaps/05%20-%20Project%20Continuity%20and%20Changing%20Facts.md#revisit-uploaded-documents-and-files-after-experimental-characterization).

Twelve questions cover focused engineering lookup, chronology/revisions, failure
diagnosis and current state, with three questions per category. Nine broad
questions reuse an earlier development set; three focused questions use existing
reference facts. Production automatic tactic selection chooses decision history
for ten and hierarchy plus neighborhoods for two. These are engineering/history
questions, not a representative sample of every possible user query.

Eight configurations produce 96 retrieval cases: baseline, six recall modules
individually, and all six together. Exact identical reader inputs share one
generation and grade, leaving 51 distinct reader requests. Every experimental
case preserves the baseline routes, hydration and automatic tactic decision.
The initial graph replays three previously observed baseline packet accesses,
once each, into a disposable database. It is a small development graph.

The reader is `codex_sdk/gpt-5.6-sol` at the authorized central-dev gateway,
with reasoning disabled, a 1,000-output-token cap and a 450-word instruction.
Twelve blinded question-level comparisons use the same model as judge against
the frozen original source quotes. Labels are shuffled; identical answers share
grades. Checkpoints score 0/1/2 for absent, partial or substantive coverage.
Question-macro and pooled scores must be distinguished. Missing exact support
quotes do not mean that a fact is absent from the summary hierarchy.

## Local retrieval timing

Times include native recall and packet construction with resident models;
they exclude startup, reader generation and learning. Configuration order is
fixed, so differences of a few hundredths of a second are not reliable speedups.

| Configuration | Mean recall (s) | Mean packet tokens | Qwen prefix passes |
| --- | ---: | ---: | ---: |
| Baseline | 1.000 | 28,977 | 0 |
| Learned associations | 0.988 | 28,977 | 0 |
| Heat diffusion | 1.033 | 28,977 | 0 |
| Qwen summary routing | 1.508 | 29,651 | 2 |
| Qwen associations | 1.127 | 29,689 | 1 |
| CAV associations | 1.155 | 29,667 | 1 |
| Live head memory | 1.644 | 29,543 | 9 |
| Combined | 2.096 | 29,577 | 13 |

The combined maximum was 2.588 seconds. Qwen summary routing is the bounded
atomic-summary tournament implemented in 265, not a new corpus-wide hierarchical
search. CAV uses the earlier layer-4 FP16 fit; its small concept-classification
score is not answer accuracy. The resident prefix is losslessly compressed FP16.

The sparse initial graph produces identical baseline packets on all twelve
questions. Qwen additions deliver between zero and four extra sections. None
increases the count of completely delivered private checkpoint quote sets:
11 question/checkpoint pairs have complete exact quotes in every configuration.
The 51 stored hierarchy summaries can still supply other checkpoint facts.

## Answer quality

Here, **baseline means the current default memory system**, not full context. It
uses automatic BGE tactic selection, the established decision-history or
hierarchy/neighborhood route and exact hydration. The prior full-context control
was not rerun or scored in this experiment.

All 51 reader requests completed with a normal stop receipt under the requested
Sol model. Reported use was 1,513,979 reader prompt tokens plus 26,545 completion
tokens. The twelve judging calls brought the complete campaign to 1,562,022
prompt and 29,638 completion tokens. No answer cited a label absent from its own
packet. A separate audit confirms that all 96 packets retained every exact
baseline reference.

| Configuration | Audited macro | Raw judge macro | Audited points | Wins / ties / losses vs baseline |
| --- | ---: | ---: | ---: | ---: |
| Baseline | **75.42%** | 75.42% | **60 / 84** | — |
| Learned associations | 75.42% | 75.42% | 60 / 84 | 0 / 12 / 0 |
| Heat diffusion | 75.42% | 75.42% | 60 / 84 | 0 / 12 / 0 |
| CAV associations | 70.14% | 70.97% | 57 / 84 | 1 / 8 / 3 |
| Live head memory | 68.26% | 68.26% | 55 / 84 | 1 / 7 / 4 |
| Qwen associations | 67.92% | 68.75% | 55 / 84 | 0 / 9 / 3 |
| Qwen summary routing | 67.43% | 68.26% | 54 / 84 | 1 / 6 / 5 |
| Combined | 66.88% | 66.88% | 54 / 84 | 0 / 8 / 4 |

The graph modules tie because the three-access graph supplies no additions in
this initial reader assay. The active selectors add roughly 566–712 packet
tokens on average without covering a new frozen support quote: every arm covers
the same 35 of 92 exact quotes and the same eleven complete checkpoint sets.
More text is therefore not more evidenced coverage in this run.

Focused lookup is already saturated: baseline scores 100%, while every active
selector loses the no-reheat evaluation detail on one question and scores 83.3%.
Failure diagnosis is unchanged at 63.3% for CAV and degrades to 56.7% for the
other active selectors on the question that requires the mixed-granularity
failure. Chronology/revision is strongest at baseline (81.7%). Qwen association
and summary routing find relevant T403 on the architecture-evolution question,
but their apparent one-point wins were scoring errors: both answers still omit a
required delivery detail. They lose more detail on the broader inventory.

Current-state questions provide the only positive directional signal. Baseline
scores 56.7%; live-head scores 62.2%, while CAV and summary routing score 58.9%.
Live-head has one win and no losses; CAV and summary routing each have one win
and one loss. This is not yet evidence that additions caused the gains. On Q09, additions include
tool observations and terse process text such as “fixing both.” On Q15, CAV
additions are code/test fragments. None overlaps a reference support quote, yet
single generations phrase existing packet evidence differently. These results
are a detector or second-pass hypothesis, not a promoted retrieval policy.

The combined arm is worst overall and produces no wins. Round-robin mixing spends
thirteen Qwen prefix passes but does not combine independent proof; it often adds
multiple progress/tool fragments. Selector fan-out should not be enabled on this
evidence.

The human audit changes three of 336 arm/checkpoint scores. Qwen association and
summary routing lose full F21 credit on Q05 because each omits a required delivery
detail; CAV loses full F04 credit on Q15 because it never identifies the allocator
fix. This removes both Q05 wins and one CAV current-state win. The raw grades are
retained, and `scoring-adjudication.json` records every override and rationale.

The blind judge also flags a shared wording error on the timestamp-loader answer:
the 2.7% to 18.9% measurement was answer presence in the ingested haystack, not
retrieval-stage presence. A Q07 diagnostic says pooled-span replication was
pending; the record distinguishes the replicated pooled-span result from a
different stratified result, so that diagnostic is not used as a causal module
finding. Point scores remain the frozen automated result, with these manual
qualifications recorded rather than silently changed.

## Characterization and decision

- **Learned associations:** use after repeated real, successful co-accesses have
  matured the graph. The tiny graph cannot help immediately. It is the cheapest
  graph reader and remains a targeted experiment, not a cold-start tactic.
- **Heat diffusion:** use when useful evidence is more than one learned link away.
  It begins returning evidence after the nine-observation learning replay, but
  its answer value remains untested. Direct associations should remain first;
  diffuse only when the direct frontier is weak.
- **Qwen summary routing:** the best fit is missing-facet or current-state
  recovery over a BGE shortlist. It can find a relevant late checkpoint, but the
  additive policy distracts focused and broad answers. It needs a gap trigger and
  stronger rejection of tool/progress summaries.
- **Qwen associations:** the fastest active Qwen selector and the most plausible
  one to retest. It helps one architecture-evolution question and is otherwise
  neutral or worse. Prefer it as a bounded retry, not every recall.
- **CAV associations:** is the strongest active selector overall, but its selected
  passages do not explain the isolated current-state win and its concepts target
  broad context and binding. Refit on evidence utility/currentness before
  production routing.
- **Live head memory:** has one current-state gain but costs nine passes and loses
  four questions. Its recurrent workspace shows no advantage over one-pass Qwen
  association here.
- **Combined selector:** disable. It is slower and less accurate than the default
  and graph-only configurations in this assay.
- **Qwen consolidation:** keep opt-in. It changes durable scalar masses, but five
  of six held-out packets equal ordinary rank learning and its one changed heat
  passage is not clearly better. Rank learning is eight times faster and remains
  the supported default.

The next useful experiment is narrow: detect an evidence gap after the default
tactic, run one Qwen-association retry, reject tool/progress-only summaries, and
hydrate only a passage that supplies a missing facet. A mature success-linked
graph should then be evaluated separately. Increasing additions, stacking
selectors or weighting every learning event is not supported by these results.

## Learning replay

A separate, local replay compares ordinary rank-based learning with
`qwen_consolidation` over the nine non-current-state questions' observed baseline
deliveries. It uses two disposable copies of the original database and the real
`learn_native_packet`, `summary_weights`, and consolidation persistence functions.
Each question contributes one observation; duplicate event retries are verified
not to learn twice. No answer or grading labels affect graph construction.

The nine short input questions receive exact identity summaries; existing source
summaries are reused. This exercises learning with captured original chunks but
does not run a full streaming ingestion/publication cycle. The subsequent three
current-state questions are withheld from learning. The original archive remains
unchanged.

- Rank updates average **0.0196 s**; Qwen-weighted updates **0.1627 s**, excluding
  model preparation. Each weighted update takes one prefix pass.
- The sixteen-node learning cap reserves one input and fifteen source chunks.
  Qwen inspects eight source candidates per update; seven retain rank weights.
  The input also retains its rank weight. This is bounded partial reweighting.
- Rank and Qwen stores retain 763 and 760 edges respectively. Of 729 shared
  edges, 589 have different accumulated masses. Published summary graphs contain
  16 and 17 admitted edges respectively, across thirteen summary nodes.
- After these additional observations, direct associations add one section for
  Q09 and Q12 and none for Q15. Heat adds one, one and two respectively.
- Five of six rank-versus-Qwen follow-up packets are exactly identical. The
  sixth, Q15 with heat, swaps a passage: rank selects T225's correction of the
  day-decay abstraction; Qwen selects T180's earlier cross-turn chunking results.
  Both also add T386's earlier 231-test completion report. No reader comparison
  was run on these learning packets, so the swap is not an accuracy gain.

Learning can therefore change and persist scalar links cheaply, and more
observed accesses can make the previously inactive graph contribute passages.
This replay does not establish that Qwen weighting improves future answers.
Historical completion claims also need chronology-aware reading: a passage can
be authentic while describing a state later corrected.

## Evidence and reproduction

- `eval_results/native-experiment-quality-20261008-r1`: frozen plan, 96 packets,
  deduplicated reader jobs, requests/responses, source-bound references, raw
  `answer-report.json`, and transparent `scoring-adjudication.json`.
- `eval_results/native-learning-quality-20261008-r1`: matched disposable learning
  databases, per-update receipts, held-out packets and `report.json`.
- `tools/characterize_native_experiments.py`: prepare, first, generate, grade,
  report phases. Responses are reused only for identical requests; an ambiguous
  reserved request is never automatically resent.
- `tools/characterize_native_learning.py`: local learning replay; accepts
  `--assay` pointing to the frozen reader assay and a fresh `--root`.
- `tests/test_characterize_native_experiments.py`: three passing guards for exact
  quote interval coverage, complete-input deduplication and strict judge coverage.

Single generations, three questions per category, selective reference
checkpoints, one development archive and the same reader/judge model limit any
ranking. A mature live graph, another domain and uploaded files remain untested.
