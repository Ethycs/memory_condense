# Matched engineering and research battery

**Status:** Complete — all 20 pairs attempted and graded; failures retained.  
**Date:** 2026-09-25.  
**Protocol:** [Canonical battery guide](../../evals/engineering_research/README.md).  
**Run:** [Frozen plan and live artifacts](../../eval_results/engineering-research-live-20260925-r3/run-plan.json).

The battery compares the same `codex_sdk/gpt-5.6-sol` actor with full historical
context versus application memory on 20 artifact-producing checkpoints from ten
real engineering/research session families. Natural historical prefixes span
4,189–98,809 tokens. It is a continuation/artifact comparison, separate from the
million-token QA campaign. The four development and sixteen validation cases
were author-inspected; they are not an unseen test population. Related checkpoints
within each family are not independent session samples.

The strongest result is engineering: both arms produce all requested artifacts,
all 14 generated code implementations pass their own tests, and all 12 with
fixed-interface independent checks pass those checks. Memory uses **85.7% fewer
actor input tokens** across the eleven pairs where both arms finish, excluding
ingestion and grading. Research exposes specific evidence-delivery defects;
ingestion failures and long-answer timeouts prevent eight of ten paired
research-quality comparisons.

| Completed outcome | Memory | Full context |
|---|---:|---:|
| Engineering artifacts present | 10/10 | 10/10 |
| Engineering arms finished | 9/10 | 10/10 |
| Research artifacts present and arms finished | 5/10 | 5/10 |
| Total requested artifact sets present | 15/20 | 15/20 |
| Total arms finished | 14/20 | 15/20 |
| Final memory reopen verified | 14/20 | Not applicable |

All fourteen finished memory arms pass final ingestion and separate-process
reopen. Initial ingestion and reopen succeed for seventeen of twenty memory
arms, including cases whose later answer generation times out.

## Engineering results

All ten engineering pairs produced their requested artifacts. All **14 code
implementations pass their own unit tests**, and all **12 implementations with
fixed-interface independent behavioral checks pass those checks**. The two E07
interface examples also run under the restricted executor and demonstrate
accepted, incompatible, and experimental compositions.

These observations do not make every execution a complete success. E07 memory
received repeated empty gateway replies, exhausted the original action limit,
and failed final ingestion because those empty replies were represented as empty
conversation turns. Its code exists and passes tests, but the run is unfinished
and its final memory lifecycle is unverified. The full-context arm also received
empty replies before eventually finishing. This is an operational failure, not
evidence of a difference in code correctness.

The source/artifact review also identifies smaller differences:

- E02 full context makes the requested VS Code integration optional.
- E03 full context sums reverse-edge duplicates, which can change merge
  priority; a separate executable diagnostic confirms the effect. Memory's
  implementation is invariant on that diagnostic.
- E06 memory retains 5–7 sentence bounds but omits the explicit downstream
  instruction that the report explain **how** to integrate nodes.
- E08 does not fully carry the historical experimental-compatibility policy
  into its documentation: partial mention in memory, exclusion in full context.
- E04's shared connectivity-discussion omission and E07's ordered-composition
  interpretations expose rubric-scope ambiguities; they are not failed code checks.

On source-reviewed engineering criterion totals, memory leads on three tasks
(E02, E03, E08), full context on one (E06), and five tie. E07 remains unresolved.
Of 40 engineering criteria per arm, memory has 36 met, three partial and one
unresolved; full context has 35 met, three partial, one unmet and one unresolved.
These artifact ratings are separate from E07 memory's incomplete execution.

Original strict scores are retained. A separate offline assessment records exact
artifact spans for reviewer quotation errors and labels manual adjudication as
unblinded. It does not silently substitute a favorable review when another
review disagrees.

## Research and execution findings

R01, R02 and R03 memory fail before answering when raw-summary support does not
pass exact bounded-quote validation. R01/R02 copy line-numbered excerpts without
preserving embedded line markers; R03 also encounters Markdown/whitespace
differences. All three allowed attempts on their blocking batches fail. The cases
remain in the denominator and their full-context controls still run. R01/R02
share a source family.

A separate offline diagnostic maps formatting-equivalent quotations back to
literal source spans, preserving numbers, words, punctuation and the original
summary. Long restored quotations are split into exact pieces under the existing
32-token support limit. It recovers **8 of 10 saved failed summary responses**,
including the terminal responses that blocked all three cases. It changes no
live policy or cache and makes no provider calls. This demonstrates a concrete
format-handling repair opportunity; it is not a recovered task-quality score.

R04 memory finishes, verifies all 28 source citations, and reopens successfully.
It meets three of four semantic criteria but omits the updated failed endpoint:
57.23% versus a frozen 60% requirement. The selected summary explicitly contains
that comparison. Hydration drops its 2,048-token raw section because the section
plus framing exceeds the 2,048-token context budget. A retained later section
contains 57.23% and the broad scientific limits, but not the failed threshold.
This is a demonstrated delivery gap after routing, not an absence of all newer
evidence. The full-context control times out before producing an artifact, so
this case does not establish a paired quality advantage.

R01–R03 full-context citation inspection finds twelve non-verbatim quotations
whose underlying propositions are present in the named sources. Most render
LaTeX notation or omit math delimiters; one also makes minor wording changes.
They still fail the frozen exact-citation requirement. This is distinct from
inventing the cited proposition, and does not certify the historical science.

R06 meets all four semantic criteria in both arms. Memory additionally passes
all 29 exact source citations and completes final ingestion and reopen. Full
context has 29 exact citations and nine that omit Markdown emphasis.

Seven actor requests time out under the configured 240-second limit: both arms
of R07 and R08, plus full context in R04, R09 and R10. These are absent outputs,
not observed incorrect research arguments. Short summary calls continue returning
normally. No failed actor request is silently retried.

R09 memory finishes with three criteria met and one partial, all 21 citations
exact, and a verified final reopen. It omits the user's RES definition and says
the supplied record does not expand the acronym. The short definition at T0037
and the global-relation premise at T0039 are absent from both selected routes
and hydrated evidence; the actor does not request recall. This is a routing gap,
plus an overbroad inference from missing retrieved evidence, distinct from
R04's hydration-budget failure. R09 full context times out before producing an
artifact, preventing a paired content-quality comparison.

R10 memory finishes with two criteria met and two partial, all seventeen
citations exact, and a verified final reopen. Its distinction between null
measure, small probability and cardinality is sound at the level assessed here.
However, it says no ambient space or measure was specified, while source T0000
contains a proposed `(Ω, F, µ0)` construction. That source document, the T0039
computability premise, and the T0111 realizable-goal definition are all absent
from the selected routes and evidence. The answer gives a general audit instead
of assessing that supplied construction. This does not establish that the
archived construction itself is mathematically valid.

| Research case | Memory criterion scores | Full-context criterion scores | Completion limitation |
|---|---|---|---|
| R01 | — | 2, 2, 2, 2 | Memory ingestion rejected |
| R02 | — | 2, 2, 2, 1 | Memory ingestion rejected |
| R03 | — | 2, 2, 2, 2 | Memory ingestion rejected |
| R04 | 2, 0, 2, 2 | — | Full-context timeout |
| R05 | 2, 2, 2, 2 | 2, 2, 2, 2 | Both finished; citation-format defects |
| R06 | 2, 2, 2, 2 | 2, 2, 2, 2 | Both finished; full-context citation-format defects |
| R07 | — | — | Both timed out |
| R08 | — | — | Both timed out |
| R09 | 2, 1, 2, 2 | — | Full-context timeout |
| R10 | 2, 1, 1, 2 | — | Full-context timeout |

Scores are source-reviewed criteria: 0 unmet, 1 partial, 2 met. A dash means no
artifact was produced; those attempts remain failures in the full population.
Only R05 and R06 support completed paired research-quality comparisons, and both
tie on their four criteria. Other source inspections diagnose the memory outputs
without establishing how a completed full-context answer would have performed.

## Tokens, lifecycle and scoring interpretation

The eleven completed matched pairs use **125,146 memory actor input tokens
versus 874,256 full-context actor input tokens**, a reduction of **85.7%**.
This includes all actor actions in those pairs, including imperfect outputs;
it excludes ingestion and evaluation. The eligible case IDs are in the final
overview.

Whole-campaign usage, including unsuccessful attempts and shared preparation
counted once, is:

| Operation | Input tokens | Visible/reported output tokens |
|---|---:|---:|
| Memory actor | 351,961 | 52,602 |
| Raw-summary compiler | 675,601 | 52,589 |
| Hierarchy merges | 100,454 | 24,840 |
| Full-context actor | 2,298,876 | 62,721 |
| Evaluation grader | 2,030,137 | 27,665 |

Memory actor plus preparation totals 1,128,016 input tokens, versus 2,298,876 for
the full-context actor. That campaign payload is 50.9% lower, but different failed
tasks and the empty-response episode make it unsuitable as an equivalent-work
cost-saving estimate. These are mixed provider counts and token estimates, not
billing measurements; failed-call output usage is unknown. Grading is evaluation
overhead and is excluded from the operating comparison.

The seventeen initial evidence packets average **1,770 native tokens** (range
1,341–2,043). All **229 hydrated spans** verify exactly. The 31 completed ingestion
receipts record no raw input to Qwen; every routing receipt records zero raw
reads and zero query-time Qwen passes. Qwen operates on hierarchical summaries
during ingestion. Its local six-layer prefix uses **float16 weights**, while
summary embedding matrices use float32. The separate raw-summary compiler and
answer model receive their authorized raw inputs.

Initial installs take 36.8–587.7 seconds, median 118.7, in this harness. Related
prefixes share summary caches, and separate-process reopening includes cold model
loading. These are not warm query-latency measurements. All query receipts are
initial retrievals: these bounded continuations never request intermediate recall
or trigger working-window eviction. This tests historical-context substitution
and final persistence, not a complete long-running engineering-session replay.

The unchanged strict report records memory **2 pass / 10 fail / 8 unresolved**,
and full context **2 pass / 11 fail / 7 unresolved**. That gate combines completion,
all criteria, exact citations and valid reviewer quotations. The offline assessment
separates those conditions: source-reviewed criterion sets are fully met in eight
memory outputs and nine full-context outputs, with one unresolved set per arm.
These criterion-only counts are not new end-to-end success rates. Manual source
and artifact adjudication is unblinded; quotation corrections preserve the original
review scores and are bound to exact artifact hashes.

The separate million-token QA campaign's **95.3% source-review-adjusted average**
is saved in the [scoring-bands note](../07%20-%20Status%20Reports/2026-09-25_scoring-bands-and-95-percent-average.md).
It is not the score of this artifact battery.

## Execution and next fixes

Execution amendments are sealed alongside the run: duplicate-fragment cache
handling, bounded summary-validation repair, standard-library test imports,
failed-case continuation, and skipping completed grading on resume. Completed
actor outputs remain intact. One redundant E02 grading call is retained in cost
accounting; sealed-artifact protection prevented it replacing the original review.
There are 461 reserved calls: 141 raw-summary, 157 hierarchy-merge, 123 actor and
40 grader calls, all within the frozen call budgets. The gateway returns 38 empty
E07 actor replies (22 memory, sixteen full context), and thirteen actor responses
exceed the requested 4,096-token output ceiling. These protocol deviations are
retained; requested identical output limits were not reliably enforced. The owned
generation worker is closed, with no pending reservations.

Gateway summary, merge and grading responses report zero usage counters.
Accounting therefore uses their saved prompt and output token estimates, marked
as estimates, rather than treating ingestion as free. Actor usage is measured
when the gateway supplies it. Empty gateway replies have no reported usage, so
their dispatched prompt estimates are not confirmed billed token counts.

The next changes should target the observed failures:

1. Restore formatting-equivalent support quotations to exact source spans before
   rejecting ingestion, preserving the bounded support contract. The offline
   8/10 recovery is a parser diagnostic, not a recovered task score.
2. Align raw-section size and hydration capacity, including framing, so selected
   qualifying endpoints can actually reach the reader.
3. Retrieve foundational source documents and definitions alongside the user
   spine, and request recall before declaring a referenced term or construction
   absent. R09 and R10 expose this dependency.
4. Enforce gateway output limits and investigate long-answer timeouts separately
   from retrieval quality.

The quotation, hydration and definition-routing fixes are not silently applied
to this baseline. The engineering results support useful application behavior;
the research failures identify concrete limits and the next tests to run.

## Evidence

- [Final overview and accounting](../../eval_results/engineering-research-live-20260925-r3/final-overview.json)
- [Source-reviewed assessment](../../eval_results/engineering-research-live-20260925-r3/assessment-20.json)
- [Unchanged strict report](../../eval_results/engineering-research-live-20260925-r3/report-20.json)
- [Example execution diagnostic](../../eval_results/engineering-research-live-20260925-r3/diagnostics/E07-runnable-examples/report.json)
- [Reverse-duplicate diagnostic](../../eval_results/engineering-research-live-20260925-r3/diagnostics/E03-reverse-duplicate/report.json)
- [Verified reviewer quotation corrections](../../eval_results/engineering-research-live-20260925-r3/diagnostics/review-evidence-corrections-11.json)
- [Initial source/artifact adjudications](../../eval_results/engineering-research-live-20260925-r3/diagnostics/manual-adjudications-07.json)
- [Interface and feedback-loop review](../../eval_results/engineering-research-live-20260925-r3/diagnostics/manual-adjudications-11.json)
- [Implementation amendments](../../eval_results/engineering-research-live-20260925-r3/implementation-amendments)
- [Offline raw-quote repair diagnostic](../../eval_results/engineering-research-live-20260925-r3/diagnostics/raw-quote-format-repair/report.json)
- [R04 selected evidence dropped during hydration](../../eval_results/engineering-research-live-20260925-r3/diagnostics/long-fragment-budget/R04-observed-failure.json)
- [R09 definition-routing gap](../../eval_results/engineering-research-live-20260925-r3/diagnostics/R09-definition-routing-gap.json)
- [R10 foundational-source routing gap](../../eval_results/engineering-research-live-20260925-r3/diagnostics/R10-foundational-source-routing-gap.json)
- [Final reviewer quotation correction](../../eval_results/engineering-research-live-20260925-r3/diagnostics/review-evidence-corrections-20.json)
- [Non-verbatim research citations checked against source](../../eval_results/engineering-research-live-20260925-r3/diagnostics/research-citation-source-review-15.json)
- [R02 source/artifact adjudication](../../eval_results/engineering-research-live-20260925-r3/diagnostics/manual-adjudications-15.json)

The live harness, offline accounting and quotation diagnostic pass **20 focused
tests** in the final verification.
The source preparation previously passed 45 tests and audited all 20 source
boundaries and 80 rubric criteria. These harness checks are separate from the
generated implementations' tests above. The final audit verifies all 436 effective
implementation hashes against the frozen plan and its sealed amendment chain.
