# Routed full-source repair — roadmap to the 84-question ceiling

**Status:** R0--R2 complete; numeric route accepted narrowly at net +1, R3 pending  
**Date:** 2026-08-26  
**Baseline:** fixed-S1 Terra answers, 56/100 under the independent Sol judge  
**Goal:** repair answer use after nominal source acquisition without changing
the sealed retrieval selection

## Finding that sets the order

The locked-100 retrieval-style analysis in
[Research Log 46](../10%20-%20Research%20Log/46%20-%202026-08-26%20-%20Retrieval-style%20intra%20and%20inter%20method%20diagnosis.md)
partitions the 44 errors into:

| Boundary | Errors | Meaning |
| --- | ---: | --- |
| nominal full-source S1 | 28 | selection within a source, packing, representation, or answer reasoning |
| partial-source S1 | 7 | at least one labeled session is absent |
| zero-source S1 | 9 | every labeled session is absent |

The **84/100** figure is `56 + 28`: the ceiling obtained if every nominal
full-source miss were repaired with zero regressions. It is not a forecast or
measured improvement. Even reaching that ceiling would leave retrieval work
before the 95% target.

The largest contrasting cells are sufficiently clear to force sequencing:

- 11/15 numeric-aggregation misses are nominally full-source, so their first
  treatment is fact/operand representation and calculation;
- 8/9 temporal-order misses have incomplete sources, so their main treatment
  belongs to a later date-aware retrieval roadmap;
- state update is already 13/14 and is a preservation control.

## Scope

This roadmap changes only the answer-time projection of the already sealed S1
selection. It reuses the EM v2 result from
[Research Logs 41–43](../10%20-%20Research%20Log/) rather than creating another
corpus build or retrieval stack.

In scope:

1. infer answer operation from the dated question alone;
2. preserve exact S0 and derive EM as S1 minus S0 only after S1 selection;
3. convert the selected neighborhood into exact-quote-cited, operator-ready
   facts;
4. synthesize or calculate a short answer under the existing 8,000-token
   workspace;
5. measure routed gains and regressions against the sealed baseline.

Out of scope:

- using gold, benchmark category, labeled sources, source completeness, the
  posthoc operator label, or the Sol verdict at route time;
- widening retrieval to repair the 16 incomplete-source cases;
- changing S0–S3 or overwriting any sealed baseline artifact;
- claiming that a result tuned on this analysis-used population generalizes;
- claiming external competitiveness before the matched Mem0 arm.

## Immutable baseline

| Input | SHA-256 |
| --- | --- |
| locked retrieval | `e36b54ec6171aa7b40f75682ad85e5822a64d45bc411ffe03bcd9cad0222007f` |
| sealed Terra answers | `d7fc47b8d1f372f002230c6ffe489dac8cd11bd71b35b8d3008b1255da2a38cd` |
| independent Sol judgments | `5dc56a240315c5577d1032d40429df7e39adad0f40a098abc371ee2ea2ec77df` |
| posthoc style ledger | `b96343ab63cbe5f4e28f408921dfe1839dfa36b6c5bde4c82304c80f7c1268d5` |
| flattened style CSV | `59e7991ddaf23ec39c8bc1963b0e84b064217b2591d6e9f9774a3707fa10ae07` |

The baseline must replay before a routed run can be released. New artifacts
live under a separate root:

```text
eval_results/longmemeval-1m-routed-full-source-repair-20260826/
```

## Core rule

> Route from the question. Select unchanged evidence. Deduplicate after
> selection. Transform to cited operator-ready facts. Seal predictions before
> judging.

The route is diagnostic intent translated into a gold-blind runtime policy.
It is never allowed to ask whether the current question was one of the 28
posthoc full-source misses.

## Budget isolation

One shared packing budget lets the highest-volume method suppress every later
method.  Each retrieval method therefore needs three independently reported
limits: a candidate quota, a packed-token ceiling, and a score normalized
within that method before cross-method composition.  S0 keeps a non-borrowable
control reserve; EM, episodic/Hebbian, and CAV/link evidence receive bounded
protected allocations; only unused capacity enters a final shared residual
pool.  The question-only route may shift allocations inside declared bounds,
but it may not consume the S0 reserve or attach an unbounded raw tail.

R0--R3 operate on the already sealed S1 selection, so they do not retroactively
change retrieval quotas.  They must still publish per-question compression and
answer token use, fact count, fallback status, and cap compliance.  Method-level
candidate and packed-token budgets become a release gate when the accepted
answer operators are composed with S2/S3 in R4.

## Answer-ready intermediate representation

Every transformed fact retains its existing exact quote citation and source
coordinate. The shared representation adds only route-specific structure:

| Route | Required structure |
| --- | --- |
| numeric aggregate/compare | cited operands, units, operation, exclusions |
| temporal interval | cited start/end events, dates, requested unit |
| preference synthesis | entity/aspect, dated claims, supersession/conflict |
| set/list join | one cited member per fact and a deduplication key |
| temporal order/select | one dated event per fact and requested direction |
| direct lookup | highest-density cited candidate and entity disambiguation |
| sufficiency check | required slots, present slots, explicitly missing slots |

The model may extract and normalize these fields but may not invent an operand,
date, status, entity, or value. Exact quote validation remains the mechanical
grounding gate. Citation validation proves byte presence, not entailment, so
the answer model still receives the citation text.

## Execution sequence

### R0 — Freeze and firewall

- Verify all four baseline artifacts and their population bindings.
- Replay the sealed judge result at 56/100 with zero calls.
- Create a new routed treatment identity and output root.
- Ensure the route plan contains no reference answer, verdict, benchmark
  category, evidence-source label, or posthoc style assignment.

**Gate:** exact baseline identity, 100 ordered questions, zero provider calls,
and no write to the baseline root.

### R1 — Question-only router and locked-S1 adapter

- Adapt the merged locked retrieval into the existing typed EM question view
  without relaxing its per-shard receipt validation.
- Classify the dated question deterministically into one answer operator.
- Keep the raw fixed-S1 path as the explicit control and fallback.
- Preserve S0 exactly; derive the EM delta only after validating S0 as the S1
  prefix; remove S0 duplicates by evidence identity and `(source, text)`.
- Bind route, selected evidence, prompt, and baseline prediction identities in
  a provider-free route plan.

**Gate:** deterministic routes, unchanged S1 evidence projection, exact prefix
preservation, no gold-bearing input, and byte-identical plan replay.

### R2 — Numeric repair first

- Apply the numeric route to every question classified numeric from its text,
  not only to known misses.
- Reuse EM v2 atomic cited facts, extended with explicit operand/unit retention.
- Ask the final responder to perform the named count, sum, difference,
  percentage, or comparison and return the requested answer shape.
- Do not attach the global raw EM tail. A raw fallback may contain only cited
  rows selected after fact conversion and must remain separately measurable.

**Gate:** every number in the answer-ready facts has a supporting quote;
calculation prompts fit the cap; semantic scoring reports rescued and regressed
baseline answers separately. Advance only on positive net marginal.

**Measured 2026-08-26:** passed narrowly. The question-only router selected 32
questions. Thirty-two Terra compression calls produced 19 valid fact packets,
12 empty packets, and one invalid packet; the latter 13 preserved the baseline.
Nineteen Terra answer calls changed 11 predictions, so only those 11 required
new Sol judgments. The paired semantic result moved **56/100 to 57/100** with
three rescues, two regressions, and net +1. Compression, run, and judge replays
were byte-identical with zero calls. See
[Research Log 47](../10%20-%20Research%20Log/47%20-%202026-08-26%20-%20Routed%20numeric%20EM%20repair%20result.md).

### R3 — Remaining intra-method routes

Implement and evaluate in descending observed full-source miss yield:

1. direct lookup — six candidates;
2. preference synthesis — three candidates;
3. temporal interval — three candidates;
4. insufficient-evidence check — two candidates;
5. set/list join — one candidate;
6. temporal order/select — one nominal full-source candidate;
7. state update — one candidate, but keep the strong raw path as the preserve
   control until a matched treatment proves a gain.

Each route gets its own eligible population and marginal. A route that merely
adds prompt content does not earn inclusion.

**Gate:** positive net semantic marginal per accepted route, explicit
regressions, no hidden substitution of the preserved control population.

### R4 — Bounded composition

- Compose only routes that passed their individual gate.
- If compact facts are insufficient, reinsert at most the existing bounded set
  of cited selected rows; never reattach the full neighborhood.
- Deduplicate selected evidence after selection, never before it.
- Allocate and report protected per-method candidate and token budgets before
  merging; normalize scores within each method and permit only bounded unused
  capacity to roll into the shared residual pool.
- Produce one final gold-blind prediction for every question, using the sealed
  baseline prediction for any route deliberately preserved.

**Gate:** 100 predictions, exact provenance, no duplicate evidence, hard prompt
cap, accepted route gains retained, and provider-free replay from journals.

### R5 — Measurement and handoff

1. Preflight the exact unique Terra compression and answer call population.
2. Run only that sealed population; journal every request and response.
3. Seal the combined 100-question prediction artifact.
4. Build the Sol population from question, reference, and sealed prediction
   only after predictions are immutable.
5. Replay both populations provider-free.
6. Publish overall and per-route semantic gains, regressions, abstentions,
   prompt sizes, call counts, and the original intra/inter strata.

**Stretch success:** 84/100 with zero regression.  
**Acceptance:** strictly above 56/100 with no concealed regression in preserved
routes and a positive routed marginal that justifies its cost.  
**Failure:** no net gain, ungrounded facts, prompt overflow, route leakage, or a
result dependent on oracle source completeness.

## Verification matrix

| Invariant | Verification |
| --- | --- |
| question-only routing | unit tests mutate gold/category/verdict while route stays fixed |
| post-selection exclusion | exact S0 prefix and S1-minus-S0 property tests |
| citation grounding | unknown alias, altered quote, duplicate key, unsupported field rejection |
| bounded prompts | whole-population provider-free preflight before client creation |
| method isolation | per-method candidate/token ledgers, protected S0 reserve, normalized scores |
| baseline preservation | hashes and untouched-artifact checks before and after run |
| honest marginal | paired baseline/candidate rows per route, rescued and regressed counts |
| gold firewall | gold loaded only by local scoring or independent judge construction after seal |
| replay | zero-call byte-identical artifacts from immutable journals |

## Deliverables

- routed adapter, classifier, answer-ready fact schema, and prompt builder;
- focused unit and integration tests;
- `route-plan.json`, `preflight.json`, checkpoint journals, `run.json`, and
  `semantic-judge-sol.json`, each with SHA-256 sidecars;
- tracked flattened result ledger under `docs/10 - Research Log/data/`;
- Research Log 47 recording what actually ran and measured;
- a separate inter-method temporal retrieval roadmap only after this treatment
  establishes which errors remain.

The validation100 population is already analysis-used. Any tuned gain here is
development evidence. A competitive or generalization claim requires a fresh,
untouched confirmation population.
