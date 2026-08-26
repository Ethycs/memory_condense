# Retrieval-style intra- and inter-method diagnosis

**Status:** a provider-free posthoc analysis grouped the sealed locked-100
questions on two independent axes: the topology of the labeled evidence that
must be retrieved and the operator needed to turn that evidence into an
answer. This split explains more than the benchmark's broad categories do. Of
the 44 sealed errors, **16 are at the retrieval boundary** because fixed S1 did
not contain every labeled source session, while **28 are downstream of nominal
source acquisition** and therefore point first to selection, packing,
representation, or synthesis.

No responder or judge call was made. The sealed retrieval, Terra answers, and
Sol verdicts were not changed.

## Diagnostic taxonomy

"Optimal style" is an oracle analysis label, not a new runtime router. Evidence
topology uses the benchmark's labeled answer-session geometry; answer operator
uses a deterministic question/category classification. It is therefore valid
for diagnosis on this analysis-used population but must not be supplied to a
future locked responder. A deployable router must infer its route from the
question alone.

### Evidence topology

| Topology | Deterministic meaning |
| --- | --- |
| `point` | one labeled answer session |
| `local_pair` | two labeled sessions adjacent by first-occurrence rank |
| `local_fanout` | three or more labeled sessions forming one contiguous rank block |
| `dispersed_join` | two or more labeled sessions separated by at least one other session |

The ranks are taken from each original question sample's chronological first
occurrence of each source session, before that sample is namespaced into the
locked concatenation. The four insufficient-evidence questions also retain an
orthogonal negative-reference flag; absence of a positive answer does not
imply absence of relevant sessions.

### Answer operator

| Operator | Required final operation |
| --- | --- |
| `direct_lookup` | extract one stated fact |
| `state_update` | select the latest replacement or corrected state |
| `temporal_interval` | compute an elapsed duration |
| `temporal_order_select` | find, order, and select dated events |
| `numeric_aggregate_compare` | collect numeric operands, then count, sum, or compare |
| `set_or_list_join` | combine distinct items across evidence |
| `preference_synthesis` | reconcile repeated or changing preferences |
| `insufficient_evidence` | determine that the requested conclusion is unsupported |

Keeping the axes separate matters. A question can require a dispersed join but
only a direct lookup after the right sessions arrive; another can have every
source session present and still require a difficult numeric aggregation.

## Population and outcomes

| Evidence topology | Questions | Correct | Accuracy | Abstained | all sources in S1 | literal hit in S1 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `point` | 34 | 20 | 58.82% | 8 | 30 | 15 |
| `local_pair` | 1 | 0 | 0.00% | 0 | 0 | 0 |
| `local_fanout` | 0 | 0 | n/a | 0 | 0 | 0 |
| `dispersed_join` | 65 | 36 | 55.38% | 19 | 51 | 35 |
| **total** | **100** | **56** | **56.00%** | **27** | **81** | **50** |

Topology alone does not explain the answer result: point and dispersed-join
accuracy are similar. The answer operator is much more discriminating.

| Answer operator | Questions | Correct | Accuracy | Abstained | all sources in S1 | literal hit in S1 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `state_update` | 14 | 13 | 92.86% | 0 | 14 | 13 |
| `direct_lookup` | 25 | 18 | 72.00% | 6 | 24 | 15 |
| `temporal_interval` | 15 | 11 | 73.33% | 3 | 13 | 1 |
| `set_or_list_join` | 2 | 1 | 50.00% | 0 | 2 | 1 |
| `numeric_aggregate_compare` | 23 | 8 | 34.78% | 6 | 18 | 18 |
| `temporal_order_select` | 11 | 2 | 18.18% | 9 | 2 | 2 |
| `preference_synthesis` | 6 | 1 | 16.67% | 0 | 4 | 0 |
| `insufficient_evidence` | 4 | 2 | 50.00% | 3 | 4 | 0 |
| **total** | **100** | **56** | **56.00%** | **27** | **81** | **50** |

## Inter-method diagnosis

An inter-method failure means the retrieval stack did not get all labeled
source sessions to the fixed responder. It is a conservative retrieval-boundary
label, not proof that every missing session contains a necessary turn.

The stage transitions make the current problem unusually clear:

| Transition or stage | Rows added | newly any-source | newly all-source | newly literal |
| --- | ---: | ---: | ---: | ---: |
| S0 result | n/a | 91 total | 81 total | 48 total |
| S0 -> S1 direct episodes | 1,727 | 0 | 0 | 2 |
| S1 -> S2 representative bridges | 22 | 0 | 0 | 0 |
| S2 -> S3 global closure | 2 | 0 | 0 | 0 |

S1 through S3 therefore did not demonstrate their intended source-rescue
specialization on these 100 questions. S1 spent almost the whole prompt budget
on local neighbors without rescuing a labeled source. S2 and S3 then had too
little admission budget to test bridge or global retrieval meaningfully.

`temporal_order_select` is the sharpest inter-method cell: only 2/11 questions
had every labeled source in S1, only 2/11 were correct, and 9/11 abstained. Its
first repair is a date-aware dispersed retrieval path with protected admission
budget, not a broader final-answer prompt over the same packet.

## Intra-method diagnosis

An intra-method failure means the answer was wrong even though fixed S1 held
every labeled source session. There are 28 such sealed misses, 63.64% of all
errors. This bucket includes three distinct possibilities that session-level
coverage cannot by itself separate:

1. the decisive turn was not selected even though another turn from its source
   session was selected;
2. the turn was present but flattened or obscured during packing;
3. the evidence was adequate and the responder failed to extract, calculate,
   reconcile, or answer.

`numeric_aggregate_compare` is the clearest intra-method stress cell. S1 held
all labeled sources for 18/23 questions and literal reference material for
18/23, yet only 8/23 answers were accepted. More neighborhood expansion is not
the default treatment for that cell. The next arm should convert the already
selected neighborhood into dated, cited atomic facts; identify the operands;
deduplicate them after selection; and make the requested calculation explicit
to the final synthesizer.

`preference_synthesis` is mixed: 4/6 have full source coverage but only 1/6 is
correct, and none has a literal reference hit. It needs aspect/entity grouping,
recency or update resolution, and synthesis over several facts rather than a
literal extractor.

The strong `state_update` result, 13/14, is also useful: it is a preserve cell.
Any new route should leave this path alone unless it proves a matched gain with
no regression.

### Exact error boundary by operator

The 44 misses split as follows. `Full-source` is the intra-method candidate
column; `partial` and `none` form the 16-question inter-method retrieval
boundary.

| Answer operator | Misses | Full-source | Partial | No source |
| --- | ---: | ---: | ---: | ---: |
| `numeric_aggregate_compare` | 15 | 11 | 3 | 1 |
| `temporal_order_select` | 9 | 1 | 4 | 4 |
| `direct_lookup` | 7 | 6 | 0 | 1 |
| `preference_synthesis` | 5 | 3 | 0 | 2 |
| `temporal_interval` | 4 | 3 | 0 | 1 |
| `insufficient_evidence` | 2 | 2 | 0 | 0 |
| `set_or_list_join` | 1 | 1 | 0 | 0 |
| `state_update` | 1 | 1 | 0 | 0 |
| **total** | **44** | **28** | **7** | **9** |

This is not a vague category correlation. Eight of the nine temporal-order
misses cross the retrieval boundary, whereas 11 of the 15 numeric-aggregation
misses occur after full-source acquisition. Those cells need different next
methods.

## How to use the split

The next implementation should be one lightweight, gold-blind route selector
in front of the existing cumulative prefix, followed by operator-specific
packing and synthesis:

| Predicted demand | Retrieval and representation treatment |
| --- | --- |
| point/direct or state update | retain S0; extract the highest-density cited fact; avoid automatic expansion |
| temporal order/select | reserve budget for dated global/bridge candidates; render an explicit event timeline |
| numeric aggregate/compare | keep fixed evidence; EM-convert it into cited operands; invoke explicit arithmetic/comparison |
| preference synthesis | join by entity and preference aspect; retain dates and supersession; synthesize current state |
| set/list join | deduplicate facts only after selection, then union the cited items |
| insufficient evidence | require a structured missing-evidence check before allowing abstention |

Evaluation should report two marginals for every route:

- **inter-method:** new any-source/all-source/decisive-turn recovery among the
  questions eligible for that retrieval route;
- **intra-method:** answer gain among questions whose required evidence was
  already admitted, with regressions on the preserved prefix reported
  separately.

This restores the intended linear design. Each additional layer has a target
cell, an eligibility set, and a marginal it must improve; it no longer gets
credit merely for appending many rows.

The 95% target requires both sides. Correcting all 28 full-source misses while
preserving the current successes would reach only 84/100, so at least 11 of the
16 incomplete-source misses would still have to be recovered. Conversely,
retrieval-only repair of all 16 boundary misses would reach only 72/100. The
largest first cells are the 11 full-source numeric misses for EM/operand
synthesis and the eight incomplete-source temporal-order misses for protected
date-aware retrieval.

## Limits

This is a posthoc diagnostic over an analysis-used population. Source coverage
is session-level, literal matching is incomplete for paraphrases and can hit
distractors, and the deterministic operator labels are a useful taxonomy rather
than ground truth. The per-question artifact preserves the raw labels needed to
audit or revise individual assignments. Any tuned route needs a fresh locked
confirmation population.

## Reproducible artifacts

| Artifact | SHA-256 |
| --- | --- |
| canonical per-question JSON ledger in the sealed eval directory | `b96343ab63cbe5f4e28f408921dfe1839dfa36b6c5bde4c82304c80f7c1268d5` |
| [tracked flattened 100-row CSV ledger](data/longmemeval-locked-100-retrieval-style-ledger-v1.csv) | `59e7991ddaf23ec39c8bc1963b0e84b064217b2591d6e9f9774a3707fa10ae07` |

The provider-free analyzer is `tools/analyze_locked_retrieval_styles.py`.
Given the frozen cleaned dataset, rerunning it checks the retrieval, answer,
judge, split, and dataset bindings before publishing or accepting the same
bytes. The verified replay completed in about six seconds and reproduced both
hashes with zero model calls.
