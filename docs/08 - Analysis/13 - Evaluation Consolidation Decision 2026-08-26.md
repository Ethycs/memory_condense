# Evaluation consolidation versus more mechanism work — the binding constraint has moved

**Status:** CURRENT decision analysis — the thin matched-evaluation spine is
implemented and live-tested; compact-renderer and answer-fusion diagnostics
failed their preregistered gates, so no full-100 promotion run followed.
**Date:** 2026-08-26
**Evidence:** Research Logs 22, 37, 43, 45, 46, and 49–53; Analysis 12;
current experiment entry points under `tools/` and `src/memory_condense/eval/`.
**Cost:** the original analysis and implementation checkpoint were provider-free.
The appended live-control update used exactly 100 Terra and 100 Sol calls. The
later v3, v4, and dual-answer diagnostics used 30 Terra and 30 Sol calls in
total; their full cost and claim boundaries are recorded in Research Log 53.

## Executive summary — the verdict

**Yes. The project currently needs more consolidation on the test and
evaluation side than it needs another retrieval mechanism.** The mechanism
inventory is already broad enough to expose the next useful distinctions:
causal/coverage retrieval, direct episodes, representative bridges,
artifact-global closure, Hebbian association, EM fact representation, CAV
linking, and LLM synthesis. What is missing is one fast apparatus that can run
those mechanisms separately and cumulatively without changing the control,
renderer, budget semantics, or scorer.

The strongest current evidence is not that the mechanisms have reached their
ceiling. It is that the apparatus prevents clean attribution:

- the original cumulative ladder starved S2 and S3 after S1 consumed nearly
  all available context;
- the newer isolated matrix repaired starvation by branching each mechanism
  directly from S0, but no longer represents the intended production order;
- EM changes selected information *and* its representation, CAV changes links
  without evidence membership, and routed answer operators change neither;
- historical S0, raw-S1, EM, CAV, and routed prompts do not all share one
  renderer and answer policy; and
- each new arm has accumulated its own loading, validation, journaling, replay,
  and scoring path.

This is why a simpler architecture can appear to perform better even when a
later mechanism is potentially useful. The comparison is often measuring a
different budget position, representation, or answer operator—not merely the
new mechanism.

The remedy is deliberately narrow: one immutable memory-snapshot manifest,
one typed stage boundary, one runner with isolated and cumulative modes, one
answer renderer/policy, one journal/replay shell, and one flattened scorer.
Mechanism internals should remain separate. This is consolidation of the
experimental *spine*, not construction of another universal framework.

The subsequent renderer diagnostics sharpen this conclusion. On the ten
legacy/v2 verdict flips, compact question-last v3 scored 4/10, compact
question-sandwich v4 scored 5/10, and a gold-blind Terra resolver over the
same v4 evidence plus both sealed answer hypotheses scored 3/10. The sealed
legacy/v2 candidates contain a correct answer for all ten rows, but that
10/10 diagnostic union—and its 60/100 full-population counterpart—is a posthoc
oracle ceiling, not an achieved oracle-free score. The remaining problem is
therefore not just prompt packing. It includes reliable evidence-to-answer
arbitration without answer-hypothesis anchoring.

## The current evidence ledger

Only three arms presently have matched, locked-100, isolated semantic outcomes.
The other results are useful, but they answer different questions.

| Mechanism | Actual intervention | Best current evidence | What it establishes | What it does not establish |
| --- | --- | --- | --- | --- |
| S0 control | causal/coverage retrieval, packing, and common answering root | legacy renderer **57/100**; fresh common v2 **53/100** on identical retrieval; v3/v4 scored 4/10 and 5/10 only on the selected verdict-flip diagnostic | the exact sealed root is usable, and renderer/answer policy is a material treatment | a full-population v3/v4 score, pure retrieval recall, direct-only retrieval, held-out confirmation, or 95% |
| raw S1 | S0 plus direct episode membership | **56/100** external answer anchor; 1,727 added rows produced two new literal hits and no new any/all-source recovery | indiscriminate local expansion is inefficient on this population | a clean S0-to-S1 marginal; historical renderers are not exactly matched |
| EM facts | select the complete S1 delta, deduplicate S0 *after* selection, then express the delta as cited atomic facts | **60/100** versus S0 57; eight rescues, five regressions | this exact selection-plus-representation bundle is positive overall | pure episodic retrieval gain or uniform benefit; direct extraction regressed by one |
| representative bridge | add cross-session representative evidence | historical S2 added 22 rows on locked 100 with no source/literal rescue; new isolated arm pending | the old cumulative tail added no measured value under its inherited cap | that bridge retrieval itself is ineffective; S2 was budget-starved |
| artifact global | add distant artifact-wide evidence | historical S3 added two rows on locked 100 with no source/literal rescue; new isolated arm pending | the old cumulative tail was effectively a no-op | a fair test of global retrieval; the layer had almost no admission budget |
| Hebbian | add a learned co-access neighbor | old dev10 replacement arm fell from 6/10 exact and 0.836 F1 to 5/10 and 0.736, and could evict decisive evidence; robust additive locked-100 arm pending | permissive one-shot replacement is unsafe | the intended preserve-S0, robust additive arm's semantic value |
| CAV links | add a bounded concept-link guide over unchanged S0 evidence | **53/100** versus S0 57; two rescues, six regressions | the current text-consumed link guide is negative for this bank, responder, and population | a retrieval result, canonical post-ladder CAV, or full latent `X1` reinjection |
| routed numeric operator | hold raw-S1 retrieval fixed; compress operands and change the answer-time operation | **57/100** versus its raw-S1 56 control; three rescues, two regressions | representation/answer policy can improve a fixed packet | any retrieval improvement or a marginal against the exact S0/EM/CAV matrix |

The exact matched S0/EM/CAV call populations, paired outcomes, and replay hashes
are recorded in
[Research Log 50](../10%20-%20Research%20Log/50%20-%202026-08-26%20-%20S0%20EM%20and%20CAV%20isolated%20retrieval%20results.md).
The raw-S1 result and its claim boundary are in
[Research Log 45](../10%20-%20Research%20Log/45%20-%202026-08-26%20-%20Locked%20100Q%20semantic%20gate%20result.md),
and the cumulative S0–S3 retrieval counts are in
[Research Log 43](../10%20-%20Research%20Log/43%20-%202026-08-26%20-%20EM%20v2%20result%20and%20locked%20100Q%20retrieval%20merge.md).
The routed numeric result is deliberately kept in the separate answer-operator
family documented by
[Research Log 48](../10%20-%20Research%20Log/48%20-%202026-08-26%20-%20Isolated%20routed%20mechanism%20matrix.md).

Two numbers need explicit protection against accidental promotion:

1. **84/100 is not an observed score.** It is the counterfactual ceiling
   obtained by repairing all 28 fixed-S1 errors that occurred after nominal
   full-source acquisition while preserving all 56 successes. Retrieval-only
   repair of all 16 incomplete-source errors would reach only 72/100. The
   decomposition in
   [Research Log 46](../10%20-%20Research%20Log/46%20-%202026-08-26%20-%20Retrieval-style%20intra%20and%20inter%20method%20diagnosis.md)
   proves that both retrieval and evidence-to-answer conversion need work; it
   does not report an 84% run.
2. **The 263-target owner plan is an evaluation denominator, not a router and
   not measured recall.** Its primary-owner labels are analysis-only. Runtime
   mechanisms may record discovery against it only after retrieval and answers
   are sealed. A candidate union can never define the desired-memory universe,
   because undiscovered targets would disappear from the denominator.

## Why the binding constraint moved

Earlier in the project, mechanism availability was the bottleneck. That is no
longer true. There are now working implementations or adapters for every
retrieval layer and for the text-facing representation/linking path, even
though some remain unscored at the correct boundary. True latent `X1`
consumption remains a CAV mechanism gap; it is the exception, not a reason to
keep cloning whole experiment harnesses. The limiting question has changed
from “can the mechanism run?” to “can its marginal be measured quickly without
also changing three other things?”

The repository shape confirms that shift. At the 2026-08-24 V0 measurement,
`src/memory_condense/eval` contained **81,858 lines**, while `search +
associations + domain` contained **39,919**. Evaluation was already **2.05×
the size of the system it measured**. The scoped non-slow suite took 35m36s,
and an unscoped invocation walked `eval_results/` and failed collection. See
[Analysis 12](12%20-%20Verification%20Relocation%20Map.md).

The matched campaign added another visible layer of bespoke orchestration. A
spot count of nine current run/judge modules is 11,766 lines:

```text
S0 control runner
EM arm runner
CAV arm runner
Hebbian arm runner
representative/global runner
generic arm judge
cumulative retrieval
cumulative synthesis
cumulative semantic judge
```

Line count is not itself a defect. The defect is that these files repeatedly
own the same lifecycle—population loading, S0 reconstruction, prompt sealing,
provider execution, journal replay, gold opening, and publication—while each
declares a slightly different row type and validation boundary. This makes a
new mechanism expensive to add, makes a failed preflight hard to localize, and
makes “same control” a property to re-prove in each script.

The apparatus is also more exacting than the present development objective
requires. Immutable journals, byte-identical replays, and gold firewalls are
valuable publication boundaries. Running the full proof path during every
mechanism iteration is unnecessary. The correct response is not to remove
those guarantees; it is to separate the fast development lane from the
publication lane.

## Why retrieval appeared to get worse as the stack grew

There are four different causes, and a consolidated runner must keep them
separate.

### 1. Shared residual-budget starvation

The canonical cumulative experiment is easiest to state by role:

```text
membership:     S0 causal/coverage
                → S1 direct episodes
                → S2 representative bridges
                → S3 artifact-global closure
                → robust Hebbian append when eligible
representation: EM facts over the selected S1−S0 neighborhood when eligible
linking:        CAV over the final admitted evidence representation
answer:         one LLM synthesis over the assembled packet
```

EM is a post-selection representation of the S1 delta, not a second
retriever, and it does not erase the raw membership state needed by later
retrieval stages. This ordering preserves the design principle that each
layer adds greater complexity after the previous one. But its first
implementation used one nearly exhausted context allowance. On dev10, S1
added 171 rows, S2 added only five rows on two questions, and S3 added none;
on locked 100, the transitions were +1,727, +22, and +2 rows. The later
methods were present in the program but scarcely present in the prompt. Their
zero gain is evidence about budget allocation, not a fair negative result
about their specialization. See
[Research Log 22](../10%20-%20Research%20Log/22%20-%202026-08-21%20-%20Recall-guarded%20cumulative%20retrieval.md)
and the inter-method table in Research Log 46. The parent-preserving linear
contract is recorded normatively in
[DR-0035](../11%20-%20Codex%20Workstream/decisions/0035-relock-linear-cumulative-design.md).

### 2. Isolation changed the graph of the experiment

The matched mechanism matrix correctly repaired that confound by giving each
method a protected budget and testing it as a direct child of S0:

```text
            ┌─ EM facts
            ├─ representative bridge
S0 control ─┼─ artifact global
            ├─ Hebbian
            └─ CAV links
```

That star is the right design for causal isolation. It is not the intended
production pipeline. In particular, the current CAV result applies a text
link guide over unchanged S0; it does not place CAV after the complete
retrieval/representation packet. The accepted architecture instead defines
CAV as a linking/fusion layer after evidence collection, with extraction
`N→K`, reinjection `K→N`, and downstream consumption of the enriched `X1`.
The text-only responder still lacks that true `X1` consumer. See
[the graph-transformer CAV summary](../00%20-%20Theory/graph_transformer_cav_summary.md)
and
[DR-0040](../11%20-%20Codex%20Workstream/decisions/0040-cav-as-linking-fusion-layer.md).

The correct test program therefore needs *both* graphs: a star for isolated
marginals and a line for accepted composition. One must not substitute for the
other.

### 3. The layers do different jobs

S0, S1, S2, S3, and Hebbian alter raw evidence membership. EM operates after
episodic selection and changes representation. CAV links already admitted
evidence without adding a row. A routed numeric policy changes the final
answer operation. Treating all four as “retrieval arms” makes the score
uninterpretable.

The EM result illustrates the point. Relative to S0, 60 versus 57 measures the
combined value of the S1-selected delta and cited-fact representation.
Relative to raw S1's external 56, it is evidence that representation can make
the same selected neighborhood more usable. Neither comparison isolates a
pure retrieval effect.

### 4. The final prompt is part of the treatment

Evidence membership alone does not define an arm. Ordering, aliases, dates,
fact formatting, link guides, answer instructions, token caps, and fallback
language all affect the responder. The current roadmap already warns that the
historical S0, EM, and CAV templates are not exactly matched. Until one common
renderer owns typed slots for these differences, cross-arm deltas remain more
fragile than their hashes suggest. See
[Research Log 49](../10%20-%20Research%20Log/49%20-%202026-08-26%20-%20Matched%20retrieval%20mechanism%20matrix%20roadmap.md).

## What “evaluation consolidation” should mean

The consolidation should be a thin shared spine with stable boundaries. It
should not merge the algorithms or force every mechanism to emit the same kind
of payload.

### One immutable memory snapshot

The repository already has a common physical substrate: authoritative turns
and chunks plus episodic, causal, Hebbian, and other derived views in and
around the schema-v11 store. What it lacks is one logical revision manifest
that says which versions of those views a question read.

```text
MemorySnapshot
  snapshot_id + parent_snapshot_id
  authoritative transcript/chunk root
  SQLite and ANN identities
  causal/coverage revision
  episodic/discourse revision
  Hebbian revision
  CAV/feature revision
  policy, budget, implementation, and model identities
```

This is a vector over compatible base and overlay revisions, not a demand for
one monolithic database or one synchronous rebuild. Every arm reads the same
immutable snapshot. Experimental arms are read-only private overlays; they do
not reheat, learn, or commit merely because evaluation exposed an item.

For live operation, the same object supports a prompt-tick transaction:

```text
M_t + current prompt
→ question-only plan
→ specialist reads from one snapshot
→ select, then deduplicate and admit under owned budgets
→ render declared representations, then link the final packet
→ one provider-visible answer
→ commit the completed turn and eligible learning exactly once
→ M_(t+1)
```

Bounded internal selector, compression, or feature-model calls belong to
stage preparation; “one answer” means one final user-turn answer call, not a
ban on those declared internal operations.

The current user prompt is query material, not historical evidence available
to its own retrieval. A failed answer publishes no child snapshot. Hebbian
and consolidation learning observe only the final evidence actually shown in
a completed tick. Evaluation can branch many read-only arms from `M_t`, but it
must not pretend that those counterfactual branches are production turns. The
read-before-write foundation already exists in
[Prompt-driven systems consolidation](../00%20-%20Theory/03%20-%20Prompt-Driven%20Systems%20Consolidation.md)
and
[Operating Requirements R3](../01%20-%20Design/02%20-%20Operating%20Requirements.md);
the missing piece is the shared snapshot/tick receipt.

### One typed stage boundary

A common adapter should standardize only the envelope:

```text
discover(snapshot, question, owned_budget) -> CandidateSet
admit(parent_packet, candidates)            -> AdmissionDelta
represent(packet)                           -> RepresentationDelta
link(packet)                                -> LinkDelta
answer(rendered_packet)                     -> AnswerReceipt
observe(completed_tick)                     -> PersistentDelta
```

Every delta records the question and snapshot identity, parent arm, mechanism
role, candidate population, pre-dedup selection, post-dedup admission,
provenance, protected and actual tokens, fallback, and final payload identity.
The payload types remain distinct:

| Role | Examples | Legal effect |
| --- | --- | --- |
| membership | S0, direct episode, representative, global, Hebbian | add raw evidence coordinates; never evict the protected parent |
| representation | EM facts | transform an already selected delta into cited facts; discovery credit survives post-selection dedup |
| linking | CAV | add relation/link payload over fixed membership; report zero evidence additions |
| answer operator | numeric reduction, timeline, state resolution | change calculation or synthesis policy; score in a separate experiment family |
| observation | reheating, Hebbian reinforcement, causal binding | update durable state only after one completed production tick; evaluation stays read-only |

Mechanism-specific candidate generation, scoring, normalization, model
loading, topology, compression prompts, link extraction, and overflow behavior
remain behind the adapter.

### One runner, two execution modes

The same adapters should be composable into two explicit plans:

```text
isolated:    S0 ├─ A
                ├─ B
                └─ C

cumulative: S0 → accepted A → accepted B → accepted C
```

In isolated mode, every arm gets its own non-borrowable allowance and compares
against the same S0 packet. In cumulative mode, each child retains the exact
ordered parent evidence, owns a declared incremental allowance, and becomes a
no-op on failure or overflow. CAV is recomputed over the final admitted
membership; it is never copied from its S0-only ablation. Only preregistered
positive mechanism-by-information cells enter composition.

Separate budgets are therefore necessary, but they are not the whole answer.
The runner must enforce both a method-local allowance and the final prompt
ceiling. An unused allowance in one isolated arm must not be silently borrowed
by another; a later cumulative policy may explicitly reallocate budgets only
as a separately named treatment.

### One renderer, answer policy, and scorer

The final prompt should have named typed slots rather than a different template
per tool:

```text
system policy
dated question
protected raw evidence
admitted raw additions
cited fact representation
link guide
declared answer operator
```

Empty slots disappear; their order and token accounting do not. This keeps
the answer model and base instructions common while making the mechanism's
necessary representation visible and chargeable.

One scorer then joins retrieval and answer outcomes in two planes:

- **inter-method:** desired-target discovery, any/all-source recovery,
  decisive-turn or literal recovery, and admission after budget/dedup;
- **intra-method:** semantic answer gain when the needed evidence was already
  present, including rescues and regressions by question-only demand.

The immutable target-owner plan and evidence topology join only after answer
and retrieval artifacts are sealed. Runtime routing may use a classifier
derived from the question alone; it may not use benchmark labels. A single
flattened ledger should expose every candidate, selection, exclusion,
admission, prompt, prediction, score, token, local-model call, and provider
call without reconstructing those facts from six artifact formats.

## A faster operating model

Exact publication validation and rapid mechanism iteration should be separate
lanes over the same contracts.

| Tier | Purpose | Required work |
| --- | --- | --- |
| T0 pure contracts | catch structural mistakes immediately | snapshot lineage, role legality, ordered prefix, no re-admission, select-then-dedup credit, budget arithmetic, idempotency; deterministic fixtures only |
| T1 adapter conformance | prove a mechanism fits the spine | fake candidates/models, gold absence, read-only store, fail-safe no-op, typed payload and renderer checks |
| T2 fast benchmark development | answer the retrieval + summarization question quickly | cached common snapshot, 1–10 representative questions or an eligible slice, behavior seals, local or provider synthesis only when needed |
| T3 matched mechanism evaluation | estimate the real paired marginal | full chosen benchmark population, isolated and cumulative modes, common renderer/operator, changed-prediction judging, unified ledger |
| T4 publication/confirmation | support an external or 95% claim | untouched population, immutable journals, complete replay, gold firewall, exact call accounting, independent judge |

T0 and T1 should run in seconds. T2 should reuse the compiled million-token
snapshot and mechanism caches rather than rebuild the corpus. T3 pays only for
valid dependent answers and changed-prediction judgments. T4 remains expensive
by design, but runs only after a policy is frozen.

Content-addressed cache layers should be reusable independently:

1. authoritative transcript/chunks/embeddings/ANN snapshot;
2. question and exact S0 population;
3. mechanism candidate plans, histories, or feature banks;
4. selection/admission and rendered prompt projections;
5. provider request/response journals; and
6. posthoc semantic and target ledgers.

Keys must bind the snapshot, ordered questions, implementation, policy,
budget, renderer, and model/checkpoint. Arms may share immutable base and S0
artifacts; they may not share mutable quotas, residual budgets, or learning
state.

## Consolidation roadmap

The shortest useful sequence is:

1. **Seal behavior, not more interiors.** Record the current S0/EM/CAV
   population, evidence projection, prompt projection, prediction, verdict,
   and token/call ledger. Retain the load-bearing gold firewall and provider
   journals, but do not introduce another graph of same-call hashes.
2. **Define `MemorySnapshot` and typed stage deltas.** Start as evaluation
   types over existing artifacts; do not migrate the store or rewrite
   mechanisms yet.
3. **Extract the common shell.** Centralize population loading, S0 control,
   typed rendering, completion journaling/replay, changed-only semantic
   judging, and flattened scoring.
4. **Wrap, do not rewrite, the completed arms.** Reproduce S0 57, EM 60, and
   CAV 53 from existing journals with zero provider calls. Exact historical
   bytes may remain legacy artifacts; the new gate should reproduce the
   load-bearing behavior projection and scores.
5. **Attach the pending membership adapters.** Representative, global, and
   robust S0-preserving Hebbian work may prepare candidates provider-free in
   parallel, but no new answer campaign should need a bespoke runner.
6. **Run isolated marginals under protected budgets.** Report only eligible
   question cells plus whole-population regressions. A method that adds no
   evidence remains a measured no-op, not a failed campaign.
7. **Compose accepted cells in canonical order.** Preserve S0, add only
   positive membership/representation stages, recompute CAV over the final
   packet if any CAV cell eventually qualifies, then perform one final LLM
   synthesis.
8. **Use publication machinery only at the claim boundary.** A new untouched
   population is needed only when promoting a tuned result or asserting the
   95% objective—not for every local iteration.

The stop condition for the consolidation tranche is concrete: the same command
family must run isolated and cumulative plans, reproduce all three current
locked scores without new provider calls, emit one ledger schema, and accept a
new mechanism through an adapter without copying the run/judge lifecycle.

## What must not be consolidated

Over-consolidation would recreate the current problem in a more abstract form.
Keep these mechanism-specific:

- candidate topology, scoring, calibration, and selection policy;
- local feature/model/checkpoint loading and certification;
- EM compression and evidence-grounding rules;
- representative/global temporal and source geometry;
- Hebbian chronological histories, thresholds, and learning eligibility;
- CAV extraction/reinjection features and link receipts;
- mechanism-local budgets and fail-safe behavior; and
- any answer-operator prompt needed to perform a declared computation.

Also preserve the in-path behavioral guards that make the linear design real:
the parent is an ordered prefix, a child cannot re-admit a duplicate, S0 cannot
be evicted, post-selection dedup cannot erase discovery credit, evaluation
cannot mutate learning state, and a failed stage is a no-op. Those are system
semantics, not removable verification clutter.

What can move out of the hot path are redundant interior recomputations,
format-specific loader copies, and per-tool judge/replay implementations.
Analysis 12 already demonstrated the right direction: pure contract tests can
check the cumulative invariants in under a second, while boundary behavior
seals protect the artifacts that matter.

## Claim boundary

This analysis does not show that mechanism research is finished, that the
pending arms will improve recall, or that consolidation itself raises answer
accuracy. It says the next unit of engineering has higher information value
when spent on comparable and reusable measurement.

Provider-free representative/global candidate derivation and Hebbian-history
preparation can continue while the spine is extracted. New provider-bearing
mechanism campaigns should wait until they can enter that shared path. This
keeps mechanism progress moving without paying again for an experiment whose
control, renderer, or scorer cannot be compared cleanly.

## Decision

**Consolidate the test/evaluation spine first, then resume the remaining
mechanism arms through it.** The immediate goal is not a larger framework and
not exact validation of every intermediate. It is a fast memory-retrieval plus
summarization benchmark loop whose expensive proof machinery is available at
the boundary rather than imposed on every iteration.

The canonical production hypothesis remains cumulative. The isolated star is
retained as its diagnostic instrument. A common memory snapshot and typed
stage ledger are the bridge between them.

## Implementation checkpoint — Decision 2

**Decision 2 was implemented provider-free on 2026-08-26.** The implementation
is under `tools/matched_eval/`, with the command surface in
`tools/run_matched_eval_spine.py`. It deliberately leaves the sealed
S0/EM/CAV bytes and `src/memory_condense/` implementation tree untouched.

The boundary chosen here is now executable:

1. historical S0 57, EM 60, and CAV 53 remain tagged
   `legacy_renderer/*` observations and are not promoted to common-renderer
   causal comparisons;
2. every future arm starts from the same immutable snapshot and
   `matched_typed_slots_v2` renderer;
3. membership, representation, linking, answer-operator, and observation
   deltas remain distinct;
4. isolated arms own non-borrowable stage budgets, cumulative stages preserve
   the exact material parent on failure, and every accepted packet must pass
   the actual 8,000-token rendered-prompt ceiling; and
5. runtime and posthoc score data occupy separate ledgers joined against the
   complete ordered answer-row population.

The zero-call legacy migration produced 300 runtime rows and reproduced all
three scores while accounting for 362 historical Terra calls, 174 historical
Sol calls, and four shared local Qwen batches. The fresh S0-v2 preflight
produced 100/100 unique prompts with a maximum 5,525-token proxy. Its sealed
artifact is
`96c109c64fbf6232e4cfa3fbc252aa8a008624d1e1bffe29ddbf0222d8f6e315`;
the prompt-population identity is
`412b54912511fde49de02395efd3a406dff6009db323cfb4e69de16bff0eea15`.
Migration took about 1.5 seconds and the complete 23.3 MB retrieval-to-prompt
preflight took about 23.3 seconds. Neither rebuilt the corpus nor called a
model.

This satisfied the provider-free half of roadmap steps 1–4 and established
the shared adapter boundary required by step 5. Exact artifacts, commands, and
verification results for that checkpoint are in
[Research Log 51](../10%20-%20Research%20Log/51%20-%202026-08-26%20-%20Matched%20evaluation%20spine%20v2%20implementation.md).

## Live-control update — the spine worked and exposed its own treatment

The fresh common-renderer control subsequently completed with exactly 100
Terra answer calls and 100 Sol judge calls, both at zero retries, followed by
byte-identical zero-call replays. It scored **53/100 semantic**, 27/100
normalized exact match, and 0.410760 mean F1. That is below the legacy S0
observation of 57/100.

This does not reverse the consolidation decision. It validates why the
decision was necessary. Retrieval and population were held fixed, all 100
source-stage receipts matched, and all 100 provider prompts changed. Of 100
predictions, 43 remained byte-identical and received identical verdicts across
the independent judge campaigns; the 57 changed predictions produced three
rescues and seven regressions. The measured loss is therefore a renderer/
answer-policy regression, not evidence that S0 retrieval got worse.

The first common renderer moved the dated question from the generation
boundary to the start of the user message, increased mean prompt proxy from
2,604.68 to 4,492.92 tokens through a metadata-heavy surface, shortened away
specific role/approximation/temporal/calculation rules, and removed the final
`Short answer:` cue. Those are four simultaneous treatment changes.

That proposed renderer-v3 step has now been executed and rejected at the
diagnostic gate. Compact question-last v3 scored 4/10, and a question-preview
plus question-last v4 scored 5/10, on the ten legacy/v2 verdict flips. Both
full-100 populations were preflighted provider-free, but neither diagnostic
authorized a full answer campaign.

A final gold-blind dual-answer experiment then supplied the compact v4
evidence plus sealed legacy and v2 predictions as explicitly untrusted
hypotheses. Although their posthoc candidate union is 10/10 on the diagnostic
and 60/100 on the full population, Terra selected only 3/10 correctly. It
retained one of three v2 rescues, recovered two of seven v2 regressions, and
degraded two answers—ordinals 65 and 97—that v4 evidence alone had gotten
right. The posthoc packet audit found no token-cap truncation, but it did find
four clear selection/reasoning failures, two rows missing a decisive user
utterance, and one conflicting-provenance row. The new failure is therefore a
mixture of arbitration/anchoring and residual retrieval/representation gaps;
the presence of a correct answer in an alternate renderer's output does not
prove that the decisive evidence is explicit in the resolver packet.

The preregistered proceed gate required at least 8/10, all three rescues, and
at least five of seven regressions. It failed, so no full-100 v3, v4, or
synthesis answer/judge run occurred. All diagnostic answer and judge planes
replayed exactly with zero calls, 80 focused tests passed, and S0-v2 still
replayed at 53/100. The complete evidence and artifact ledger are in
[Research Log 53](../10%20-%20Research%20Log/53%20-%202026-08-26%20-%20Compact%20renderer%20and%20dual-answer%20synthesis%20diagnostics.md).

The next answer-side experiment, if pursued, must be registered as its own
operator and must resolve evidence before accepting either answer hypothesis;
naive candidate voting or ordered hypothesis presentation is not promoted.
Retrieval mechanisms can still enter the shared isolated/cumulative spine,
but their evidence-discovery results must not be confused with the 60/100
posthoc oracle ceiling. The 95% objective, untouched confirmation population,
true responder-side CAV activation reinjection, and fair Mem0 comparison all
remain open.

---

**Verification block:** this document is an analysis of already published
artifacts. Before implementation, re-check the structural baseline without
provider calls:

```powershell
pixi run python scripts/measure_verification_density.py
pixi run -e dev python scripts/dev10_replay_gate.py
pixi run -e dev pytest tests/test_cumulative_contract_invariants.py -q
```

The spot count used these paths:

```text
tools/run_locked_retrieval_mechanism_arm.py
tools/run_locked_s0_em_facts_arm.py
tools/run_locked_s0_cav_links_arm.py
tools/run_locked_s0_hebbian_arm.py
tools/run_locked_independent_closure_arms.py
tools/judge_locked_retrieval_mechanism_arm.py
src/memory_condense/eval/recall_guarded_cumulative_validation_retrieval.py
src/memory_condense/eval/recall_guarded_cumulative_synthesis.py
src/memory_condense/eval/recall_guarded_cumulative_semantic_judge.py
```

Reproduce each count with PowerShell
`(Get-Content -LiteralPath <path>).Count`. The decision does not depend on the
exact line total: the stronger preserved V0 result is that the complete `eval`
package was already 2.05× the retrieval system it measured.
