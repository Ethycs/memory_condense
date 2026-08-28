# First-principles memory stack audit: the missing layer is orchestration

**Date:** 2026-08-27

**Status:** architecture finding; provider execution paused

**Scope:** classical retrieval, decay/reheat, heat diffusion, Hebbian access
memory, adaptive transition slew, episodic memory, CAV linking, packing, and
answer operation

## Conclusion

The repository does not primarily lack another retrieval algorithm. It lacks
one runtime that owns the complete prompt tick and preserves the semantic role
of every existing mechanism.

The intended design is cumulative:

```text
discover -> admit -> represent -> link -> answer -> observe
```

The measured system drifted into two different experimental graphs:

1. a cumulative `S0 -> S1 -> S2 -> S3` line with one shared residual budget,
   where raw S1 expansion starved the later stages; and
2. a matched star whose descendants each start from S0, which correctly
   isolates marginal effects but is not the production composition.

Decay, heat, Hebbian access memory, transition slew, episodic closure, and CAV
therefore did not jointly fail. Most of them have never jointly reached the
same answer packet. Several are dormant in the current evaluation. True CAV
`X1` does have a bounded diagnostic consumer that turns its changed node scores
into an ordering receipt, but no production answer-time representation
consumer receives the enriched `X1` vectors; the responder sees only a
tensor-free order or text-link projection.

The highest-leverage missing retrieval operation is also simpler than another
graph rewrite: **select a source independently, hydrate that selected source's
full history, and map it to cited atomic facts before final packing**. In the
13 hard retrieval-boundary rows audited after the 71/100 direct-query result,
the current selected union covers 23/29 registered answer sources (79.3%) but
only 12/28 decisive answer components (42.9%). Mapping the unchanged selected
full histories raises component availability to 22/28 (78.6%) and makes 9/13
rows fully answerable. The dominant error is often the excerpt, not the source.

## First-principles contract

A memory tick should have one authoritative state transition:

```text
M_t + question_t
  -> compile a question-only intent and obligation plan
  -> retrieve protected classical anchors
  -> fan out bounded specialist candidate reads
  -> select within each specialist lane
  -> hydrate selected sources/episodes
  -> map selected material to provenance-bound facts
  -> deduplicate only after selection and mapping
  -> form episodic/temporal bundles
  -> link the final admitted frontier
  -> execute the question-appropriate answer operator
  -> observe the evidence actually exposed and the completed interaction
  -> M_(t+1)
```

This separates jobs that the experiments have repeatedly conflated:

| Mechanism | Proper role | It must not be treated as |
| --- | --- | --- |
| Classical dense/BM25/source retrieval | high-recall direct anchors and source activation | the only evidence granularity |
| Decay/reheat | long-term availability and importance prior | a hard factual-validity or supersession rule |
| Heat diffusion | within-tick exploration and source/token allocation | a replacement for strong direct evidence |
| Hebbian co-access | repeated cross-tick assembly prior | a one-shot semantic similarity score |
| Adaptive slew | prior over the next prompt's likely source/action | a terminal independent-QA retriever |
| Episodic memory | event boundaries, temporal neighborhoods, obligations, and atomic bundles | a large raw tail appended before packing |
| CAV | query-conditioned linking/fusion over the final admitted representation | another sibling text retriever |
| Answer operator | extraction, numeric reduction, timeline, set join, or synthesis | evidence discovery |

The raw transcript remains factual authority. Every compact representation,
edge, heat value, CAV, fact, or episode is a reconstructible routing or
representation layer with exact source provenance.

## What git history says happened

The implementation history is internally coherent but composition never
caught up:

| Commit | Addition or correction | Architectural consequence |
| --- | --- | --- |
| `f77781b` | stratified pooled-span retrieval | recovered evidence units split across short turns, but never entered `build_context` or the current matched packet |
| `80262ea` | typed memory, provenance, decay, hybrid retrieval | established the classical/typed base |
| `0d86038` | decay wired into ranking and bounded reheat | corrected an inert lifecycle signal |
| `9aea4cd` | wall-clock decay changed to turn-space decay | made conversation decay measurable |
| `55b3fe9` | associative graph, heat, Hebbian, transition replay | added live association mechanisms |
| `bae4bca` | prompt-driven causal consolidation | added completed-turn graph learning |
| `84943a7` | provider-free episodic discourse closure | added event/obligation closure |
| `3a6d83e` | Qwen episodic signals | added an adaptive boundary option |
| `f677eae` | deleted caller-less heat/Hebbian search wrappers | documented that live seams had no callers |
| `c683fe5` | episode-primary routing | replaced rather than extended direct authority and regressed |
| `a5e3505` | recall-guarded cumulative synthesis | restored the additive S0-S3 line |
| `6c5205a`, `a307c96`, `d1c8808` | restored and diagnosed CAV | recovered the fourth layer but retained a proxy consumer |
| `2f64626` | integrated later EM, Hebbian, CAV, and matched-arm work | supplied many mechanisms and ablations, not one prompt-tick orchestrator |

This is why the architecture can look complete in documentation and tests
while the measured answer path remains incomplete.

## Evidence from the current locked and matched paths

### Shared-budget starvation

The locked 100-question cumulative artifact uses fixed-interval episode
boundaries, a 7,000-token evidence cap, and an 8,000-token prompt cap. Its
actual stage admissions are:

| Stage | Total rows added | Questions adding zero rows | Questions reporting budget exhaustion | Mean context tokens |
| --- | ---: | ---: | ---: | ---: |
| S0 causal/coverage | 3,463 | 0 | 0 | 2,275.7 |
| S1 direct episodes | 1,727 | 0 | 0 | 6,820.9 |
| S2 representative bridges | 22 | 93 | 93 | 6,901.6 |
| S3 artifact-global closure | 2 | 99 | 99 | 6,904.7 |

S2 and S3 were syntactically present but almost never received a usable
budget. Their flat result is not a clean falsification of bridge or global
closure.

### Isolation replaced production composition

The matched population loader validates the full cumulative parent chain but
projects only `causal_graph_coverage_predecessor` into the S0-v2 population.
That is correct for an isolated-control spine. The later direct-query,
partition, guided, EM, Hebbian, and CAV experiments then became siblings of
S0 rather than successive transformations of one packet.

The strongest common matched answer result is the direct-query payload at
71/100, versus 53/100 for S0-v2. The five-arm posthoc oracle is only 74/100.
Recombining sealed final predictions cannot reach 95 because the same decisive
spans and answer operations are missing across the arms.

### Mechanisms that do not reach the current tick

- Typed-memory decay and reheat apply to `memory_items`, but the locked
  retrieval/evaluation paths generally ingest with `auto_extract=False` and
  render raw chunks. The measured packet therefore does not exercise the
  ranking behavior repaired in the decay commits.
- `expand_heat_diffusion_results` is reached by the experiment rig, not by the
  current matched or cumulative answer path. Reciprocal-rank “heat” in the
  query expansion is stateless rank fusion, not graph heat diffusion.
- `expand_hebbian` and `observe_retrieval_access` exist, but the ordinary
  matched tick has no caller that performs the read and then observes the
  final exposed set. The historical H1 experiment reconstructed access events
  offline and allowed a replacement; it lost decisive evidence and was not a
  cumulative heat-plus-Hebbian test.
- `CausalTransitionPolicy` remains a chronological replay diagnostic. It is
  not admitted to QA retrieval. LongMemEval's independent terminal questions
  also cannot measure cross-prompt slew without a separate chronological
  protocol.
- The locked episodic artifact uses `fixed_interval`, despite adaptive Qwen
  boundary code existing. Episodic reads are seeded by chunk hits because an
  independent episode-representative ANN does not yet exist.
- The latent router emits `steered_nodes` (`X1`). The fast feature-session path
  can score `X` versus `X1` and emit a tensor-free ordering diagnostic, but the
  planner then releases the tensors. The locked text responder receives a link
  guide or ordering proxy, not the enriched node representation itself.

### The current strongest arm narrows the classical stack

The 71/100 direct-query treatment is a useful query-expansion experiment, but
it is not the complete classical retrieval base described elsewhere in the
repository. `ExistingPartitionHybridSearch.search_many(...)` calls
`search_hybrid_graph(...)` with all of the following disabled:

- source slots and source-local search;
- source TF-ISF and HSC expansion;
- neighbor-radius and neighbor-slot expansion;
- the source reranker; and
- attention feedback.

It retains hybrid chunk ranking, role-aware retrieval, multi-fact source
diversity, partition routing, and ten question-derived query variants. Those
choices make the marginal attributable, but they also mean that the best
answer score to date cannot be read as the result of classical retrieval plus
all later memory mechanisms. Re-enabling each omitted classical feature needs
an isolated, direct-preserving arm before any positive feature is promoted to
the cumulative tick.

The older classical stack also contains a separate **stratified pooled-span**
retriever. It scores four- and eight-chunk spans in separate strata, then
returns their authoritative member chunks. Commit `f77781b` measured pooled
span-8 `k=3` at 21.6% recall versus 6.5% for dense `k=10` on the original
short-turn development chat, while the operating requirements record that
span beat hybrid at every matched budget in that local sweep. On the later 1M
development construction it reached 40% literal recall at 294 tokens versus
45% at 1,285 for hybrid. These are small development results, not a promotion
case, but pooled span is the obvious compact lane for facts fragmented across
adjacent short turns. `build_context` still cannot draw expansions from it,
and the current matched packet never tests it.

The existing span implementation can run read-only against the locked SQLite
embeddings without rebuilding or persisting an index, and warm scans are cheap.
Its first matched arm still needs two explicit caveats: the flattened API loses
span-level/stratum identity after returning ordinary member chunks, and current
rowid grouping can cross source boundaries or let metadata influence the pooled
score. Preserve those semantics for the first isolated arm and seal the exact
levels, returned chunks, scores, and sources; a source-bounded span variant is a
separate treatment. Pooled-span discovery should feed the adaptive source gate
before hydration, not silently masquerade as the direct lane or append an
unbudgeted fifth fact block.

Likewise, the monotone conditional information-rate filter is implemented but
absent from the matched answer packet. On the ten-question HSC4 development
run it removed 193 mean tokens (8.9%) without changing source coverage or
literal recall. That makes it a useful budget-recovery knob after protected
anchors and multi-fact guards are in place; it is not evidence for a large
recall rescue by itself.

### Earlier strong numbers were different evidence-ladder rungs

Several historical results that looked close to closure were not judged
end-to-end answer accuracy:

- the first-40 causal-consolidation preflight reported 99.5% mean registered
  evidence-source coverage and all expected sources on 39/40 questions, while
  reaching 23 of 24 haystack-literal answers;
- the chronological consolidation replay reported 38/39 literal evidence
  probes versus 35/39 for its operational parent; and
- the heat experiments reported answer-string containment at `k=5`, not a
  semantic responder score.

These are real mechanism results, but they establish source activation,
literal reach, or token efficiency. The matched 71/100 result measures judged
answers. A source can be activated while the decisive span is not packed, and
a decisive span can be packed while a numeric, temporal, set, or sufficiency
operator still fails. The measurements should therefore remain separate and
be joined only through the per-question evidence ladder.

## Obvious knobs that are still missing or misplaced

### 1. Multi-resolution source hydration

Classical retrieval should decide both *which source* and *which excerpt*.
Those are separate decisions. Source-ID recall is already much higher than
decisive-span recall, so the next child layer should:

1. preserve each method's independent source selection;
2. select sources with question-derived entity, date, and operator obligations;
3. hydrate one selected source history per bounded map call;
4. emit individually validated quoted facts; and
5. deduplicate facts across methods and against direct evidence only afterward.

The 29 registered answer histories in the hard-row audit have median 3,157,
p95 3,731, and maximum 4,131 tokens. All fit in one 8k mapping call. Across
all selected-source occurrences only three unique histories exceed 8k, so
deterministic chunk-boundary windows handle the exceptional case.

A fixed source top-k is not sufficient or economical. Across the complete
188-target registry, protected S0 covers 158 targets. Taking the top eight
distinct sources independently from direct, partition, and guided selection
raises the S0 union only to 174/188 (92.6%) while requiring a mean 17.06
unique source maps per question after cross-method caching. Uncapped, the
three methods expose a mean 146.67 distinct sources per question. The
production knob must therefore be a cheap question-conditioned source gate or
adaptive tail based on unresolved obligations and source uncertainty. It must
record each method's selection before cross-method caching/deduplication; a
wider fixed top-k merely rebuilds the million-token context as provider calls.
An exhaustive analysis-only sweep of all static `(direct, partition, guided)`
caps from zero through twelve confirms the point: the best tuple reaches only
176/188 (93.6%) at a mean 18.11 unique source maps per question. No static
tuple in that range reaches 95% source coverage even before span mapping and
answer-operation errors.

### 2. Non-borrowable admission budgets

A later method cannot demonstrate its specialization if an earlier method can
spend its allowance. Preserve S0, then give compact outputs independent lanes.
A development allocation that fits the measured 8k envelope is:

| Lane | Final compact-input budget |
| --- | ---: |
| Existing direct/two-pass packet | protected; measured maximum 5,801 |
| Direct/classical selected-source facts | 384 |
| Partition selected-source facts | 192 |
| Guided selected-source facts | 192 |
| EM facts/bundles | 256 |
| CAV link guide | 256 |
| Solver output reserve | 768 |
| Total | 7,849 |
| Safety | 151 |

Unused budget should remain unused during the first matched test. Borrowing is
a later ablation, not a default.

### 3. A question-only mechanism controller

Not every method should fire on every prompt. The controller should route from
query-derived obligations, never gold or prior judge outcomes:

| Query need | Specialist path |
| --- | --- |
| point lookup/current state | protected classical anchors, direct extraction |
| multi-event count/comparison | source hydration, fact map, numeric operator |
| chronology/latest state | dated source/episode reads, timeline operator |
| set enumeration | source-balanced mapping, set-union operator |
| explanation/preference | EM neighborhood, guarded Hebbian/heat expansion, synthesis |
| cross-item relation | CAV over the final fact/bundle frontier |
| absent/unsupported premise | exhaustive-scope witness and insufficiency decision |

The controller chooses lanes and budgets. It does not choose the answer.

### 4. Separate memory clocks

One decay number cannot express factual validity, long-term salience, current
working activation, and learned edge utility. The runtime should expose at
least four independent clocks:

- item salience decay/reheat;
- short-lived within-conversation heat;
- slower Hebbian/causal edge decay;
- event-time version/supersession authority.

Query relevance and provenance must be able to override low salience for a
rare decisive fact. Supersession is a typed temporal relation, not “old means
false.” Pins and coverage reserves remain hard constraints.

Two lifecycle parameters remain explicitly uncalibrated in the historical
design: the default 30-turn half-life under *live* reheating, and
relevance/rank-weighted reheat (the current boost treats returned ranks much
more uniformly). They are valid chronological knobs, but changing them cannot
repair an independent, read-only terminal-QA run in which the memory-item path
never observes access. They should be swept only after the evaluated tick owns
observation and the benchmark supplies repeated prompts.

### 5. Guarded heat and Hebbian use

Heat should allocate exploration and source text after strong anchors are
protected. Hebbian expansion should be additive, scoped, and earned by
repeated support:

- one or two shallow hops;
- source/session/entity scope;
- hub normalization and degree caps;
- support and co-access each at least two;
- at most one appended fact/row in the first locked arm;
- no replacement of protected evidence; and
- observation only after a completed tick, over evidence actually exposed.

These policies test the mechanisms' intended job without repeating the H1
tail-replacement failure.

The important learning distinction is *candidate*, *exposed*, and *confirmed
use*. Candidate co-occurrence must never train the graph. Exposed co-access is
the existing safe default; a later chronological arm can separately reward an
edge when a subsequent turn reuses or confirms it. Keeping those event types
separate prevents the heat/Hebbian plane from learning its own retrieval noise.

### 6. Real adaptive slew evaluation

Slew is meaningful only when `question_(t+1)` follows `question_t` against one
evolving memory. Its policy must include `stay`, split user-to-assistant from
assistant-to-user learning, and train only after the next turn is revealed.
It needs a chronological multi-query benchmark. It should not be counted as a
missing rescue in the current independent-terminal 100-question score.

### 7. Finish EM and CAV at their actual boundaries

For EM:

- compare fixed, lexical/embedding-change, and Qwen/surprise boundaries;
- add an independent episode-representative index;
- replace the certified path's exact `RuleBasedDiscourseLinker` restriction
  with a separately certified, source-validated semantic-linker arm;
- rank temporal neighbors by obligation gain per token; and
- close over compact mapped facts/bundles rather than a nearly full raw tail.

The semantic-linker point is a genuine missing mechanism rather than dormant
plumbing. The generic compilation API accepts a `DiscourseLinker`, but the
certified evaluation path rejects anything except the exact conservative rule
linker. The EM theory already calls that linker an English bootstrap and says
a stronger semantic linker remains necessary. Until that seam is measured,
revision, contradiction, dependency, and cross-episode links are systematically
underapproximated.

The safe first implementation is a sibling, source-local derived overlay bound
to immutable source-span receipts. It must not publish links into the locked
source database: that would change the bytes the fact-hydration path revalidates.
An LLM may propose only frozen evidence aliases and ontology roles; deterministic
code constructs IDs, validates every cited span/member, and admits only fully
grounded relations. Newly reached raw spans can then map into the existing EM
fact lane. The rule-linker control and its receipt remain unchanged.

For CAV:

- compute it only after final evidence/fact admission;
- use extraction `N -> K` and reinjection `K -> N` as the linking operation;
- make `X1` change the fact/link representation that the solver consumes; and
- retain the current text link guide only as a separately labeled proxy.

Changing CAV temperatures, latent count, or training loss before `X1` has a
production answer-time representation consumer tunes a signal whose diagnostic
effect still cannot reach the final operation except through a lossy proxy.

### 8. Keep answer operation separate

The 71/100 campaign still contains numeric, temporal, set, extraction, and
synthesis failures after nominal source acquisition. A fact mapper followed
by a typed solver is the right downstream layer. The sealed two-pass V2
preflight now maps evidence first and then requires `keep_parent`, `replace`,
or `insufficient`, but it should remain paused until retrieval composition is
settled. It cannot manufacture absent evidence.

## Second-pass code audit: additional live-path losses

The first provider-free composition preflight exposed several concrete losses
inside the new plumbing. These are higher priority than adding another
speculative retriever.

1. **Entity hints became mandatory obligations.**
   `query_map_source_gate_adapter.py` currently emits one SUPPORT obligation
   for every query-plan entity. Generic search hints therefore keep the gate
   unresolved even when the mapped answer is grounded. Temporal/frontier
   obligations are also intentionally impossible for the bounded V2 text map
   to close. The result is 97/100 activations. A separately sealed policy
   should use one OR-like any-anchor support obligation, at most one typed
   operation obligation, and a gold-blind parent/map disagreement trigger.
2. **Provider prompts carry receipt payload instead of compact aliases.**
   Source-history prompts repeat source IDs, chunk IDs, turn IDs, hashes, and
   offsets that the local validator already owns. Short aliases can be rebound
   after completion without weakening exact-quote, offset, or source
   provenance. The same issue occurs again when full fact origins and receipt
   hashes are rendered to the final solver under 192--384-token method lanes.
3. **Dates and speaker roles are dropped between validated layers.**
   Query expansion verifies source, role, and creation time, but the current
   direct payload and V2 map render mostly alias plus text. Source-history
   hydration still has role/time, yet the fact-origin projection does not carry
   both into the final compact frontier. Temporal and current-state operators
   therefore lose exactly the metadata they need.
4. **The direct source stream uses the older admitted delta.**
   The source-gate adapter draws direct sources from V1 admitted query spans,
   while the already sealed query-expansion repack V2 preserves a much larger
   selected-before-dedup frontier. This mainly affects adaptive tails rather
   than the first source, but it is the correct direct stream for the stated
   select-then-deduplicate design.
5. **Post-map direct dedup compares incompatible hashes.**
   A mapped atomic quote is hashed as a substring, while direct evidence is
   hashed as the whole exposed chunk. Exact-hash equality therefore retains a
   duplicate atomic fact from the same direct chunk. Containment must be proven
   against the same frozen chunk after mapping; facts from different chunks
   must remain distinct.
6. **No active adapter supplies the EM lane.**
   The fact union reserves 256 tokens for EM and can merge EM batches, but the
   composed adaptive runtime currently supplies only direct, partition, and
   guided sources. The sealed post-selection EM fact path remains a sibling,
   not the intended later layer.
7. **The historical coverage-first selector remains dormant.**
   HSC/source-local/TF-ISF behavior is disabled in the strongest direct-query
   treatment even though the remaining numeric, temporal, and set questions
   are completeness-heavy. Coverage-first ordering should be tested over the
   sealed candidate union before increasing source-map calls.

The exact strict-obligation D5/G2 preflight quantifies the multiplication:
520 unique selected sources become 1,137 unique prompt-safe history windows
and 1,106 new mapper calls under the 8k envelope. It is an upper construction,
not a runnable default. Consolidated obligations, compact aliases, compact
fact rendering, and per-policy cached hydration should be measured before any
source-map provider population is authorized.

## Recommended restoration order

1. Build one prompt-tick orchestrator and make its stage types explicit:
   `discover -> admit -> represent -> link -> answer -> observe`.
2. Repair obligation semantics, preserve role/time, compact provider-facing
   aliases, and fix same-chunk post-map dedup.
3. Keep direct S0/query evidence immutable and add an economical
   source-history hydration plus cited-fact child.
4. Wire the selected-before-dedup direct repack into adaptive source tails.
5. Restore stratified pooled-span as an isolated compact classical lane and
   retest source-local, TF-ISF, HSC, and information-rate filtering one at a
   time without replacing direct anchors.
6. Add method-local, non-borrowable budgets; select independently and dedup
   only after selection/mapping.
7. Route numeric, temporal, set, direct, and synthesis questions to distinct
   operators over the same fact schema.
8. Add the sealed post-selection EM fact lane for unresolved temporal and
   synthesis rows.
9. Reintroduce heat and robust Hebbian expansion as guarded candidate/budget
   priors, then make final-exposure observation part of the tick commit.
10. Run EM closure over compact facts with adaptive-boundary, semantic-linker,
   and independent episode-index ablations.
11. Make genuine CAV `X1` or its explicitly lossy textual projection affect the
   final linked representation.
12. Test each mechanism in an isolated star, promote only positive marginals,
   and then test the accepted mechanisms in the cumulative line.
13. Use a fresh locked population for the 95/100 claim and a chronological
   population for slew/Hebbian learning. Compare Mem0 only at the same prompt
   budget, answerer, question population, and provenance standard.

## Provider-free implementation checkpoint

Two integration contracts now make the diagnosis executable without opening a
model endpoint:

- `tools/matched_eval/prompt_tick_contracts.py` owns the ordered
  `discover/admit -> represent -> link -> answer -> observe` lifecycle. It
  requires same-parent specialist fan-out, method-local budgets, post-map fact
  deduplication, CAV over the final frontier, exact renderer exposure, one final
  answer call, and post-answer observation/commit.
- `tools/matched_eval/source_history_fact_union.py` batch-hydrates exact frozen
  source histories, excludes metadata from mapping text, validates exact-quote
  mapped facts individually, deduplicates and excludes direct evidence only
  after mapping, and packs non-borrowable direct/partition/guided/EM fact lanes
  under the 8k envelope with a separate CAV-link reserve.
- `tools/matched_eval/source_gate_controller.py` activates only from a sealed
  upstream unresolved-obligation receipt. Its default base is direct-5 plus
  guided-2, followed by small route-specific tails; it never maps all 100
  questions unconditionally. Physical work is bound to the full question and
  obligation plan. When multiple methods selected the same namespaced source
  window, one provider completion fans out through lane-specific validation so
  every method keeps logical discovery credit without duplicate calls. A
  capped ranked frontier remains unresolved unless its scope is separately
  sealed exhaustive.

Their focused suites pass 20/20 together. They are contracts and local
transformations, not a provider result. The pinned-artifact loader and live
transport remain to be connected; the current static cap sweep cannot reach
95% registered-source recall at an economical mapping budget.

## Measurement contract

Each tick should report the boundary at which a target was lost:

```text
source discovered
-> decisive span hydrated
-> cited fact validated
-> fact admitted
-> required relation linked
-> operator inputs complete
-> answer correct
```

That ladder prevents source coverage, raw literal reach, representation
quality, linking, and answer correctness from being collapsed into one score.
The development lane can use lightweight artifacts and cached replay; the
publication lane retains the existing sealed receipts, gold firewall, and
fresh held-out confirmation.

## Related records

- [Evaluation Consolidation Decision](13%20-%20Evaluation%20Consolidation%20Decision%202026-08-26.md)
- [Query answer joint failure taxonomy](14%20-%20Query%20answer%20joint%20failure%20taxonomy%202026-08-27.md)
- [System Overview](../03%20-%20Architecture/00%20-%20System%20Overview.md)
- [EM-LLM Episodic Discourse Closure](../00%20-%20Theory/05%20-%20EM-LLM%20Episodic%20Discourse%20Closure%20for%20Diffuse%20Retrieval.md)
- [Graph-transformer CAV summary](../00%20-%20Theory/graph_transformer_cav_summary.md)
- [Recall-guarded cumulative retrieval](../10%20-%20Research%20Log/22%20-%202026-08-21%20-%20Recall-guarded%20cumulative%20retrieval.md)
- [Causal Hebbian H1 restoration](../10%20-%20Research%20Log/37%20-%202026-08-22%20-%20Causal%20Hebbian%20H1%20arm%20restoration.md)
- [Matched mechanism roadmap](../10%20-%20Research%20Log/49%20-%202026-08-26%20-%20Matched%20retrieval%20mechanism%20matrix%20roadmap.md)
- [Query-era matched answer campaign](../10%20-%20Research%20Log/63%20-%202026-08-27%20-%20Query-era%20matched%20answer%20campaign.md)

## Claim boundary

This audit identifies an integration failure and a high-leverage next layer.
It does not claim that the composed stack already reaches 95/100, that every
mechanism will add accuracy when correctly wired, or that the analysis-used
100-question population is held out. No provider calls were made for this
audit.
