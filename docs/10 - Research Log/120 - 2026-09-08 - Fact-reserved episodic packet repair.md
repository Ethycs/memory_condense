# Fact-reserved episodic packet repair

Date: 2026-09-08

Status: the additive v7/r9 packet repair is sealed on the exact locked
failure30 cohort. Construction is structurally green, but the sealed targeted
accuracy is only **11/30**. Ordinal 75 now receives its missing raw numeric
neighbor, while ordinal 51 still exposes a separate latest-state semantic
residual. This is useful progress over the prior 0/30 residual, but it does not
justify a full100 promotion or a 95/100 claim.

## Question

Can the user-spine/episodic stack recover the facts that were already found by
its component mechanisms, without letting provenance text, shared episode
budgets, or optional neighborhood expansion crowd those facts out of the final
LLM packet?

The working hypothesis is now narrower than “invent a stronger retriever.” The
failure assay showed that the residual evidence was usually present somewhere
in the available memory mechanisms. The repair therefore treats packet
composition as a conservation problem: give specialized lanes independent
budgets, reserve a compact fact slice before optional episodes, hydrate exact
raw backing evidence, and run deterministic reducers only over the final
provider-visible slice.

## Immutable baseline and population boundary

The immutable parent is the v6/r3 full100 run:

- source program SHA-256:
  `24dab00ba6e7c81bfa27e443dc6f6f28eaa4aaecc8372207857d71d89926b823`;
- selection SHA-256:
  `9de07c09a8136b402158d05cef282b96d461b7baa02c0f0f239564cf39893965`;
- answer SHA-256:
  `ec40150454c91e67cffe9966efc387be1ce3c68faf37cd7d96cfbf5e17e88173`;
- judgment SHA-256:
  `e0c42a520e2cae6ecd74e0fb7a484ac0fe99e74b2dcb830110b5bc0e84e20bfa`;
  and
- judged accuracy: **70/100**.

The baseline artifact is
[the sealed v6/r3 full100 root](../../eval_results/longmemeval-1m-hot-v6-spine-episode-fact-ledger-full100-20260907-r3/selection.json).
The additive successor does not mutate that artifact or the pinned v6 source.

This is a 100-question evaluation over **ten resident, approximately
one-million-token namespaces**. The reduced cohort reuses those authenticated
resident stores and their cache/index bindings. It is not 30 independently
ingested one-million-token prompts, and no throughput or accuracy claim in this
log should be read that way.

## Exact failure30 identity

The `failure30` population is the exact ordered set of zero-based full100
ordinals judged incorrect in the sealed r3 run:

`5, 6, 14, 15, 17, 25, 36, 40, 42, 43, 48, 49, 51, 52, 53, 59, 61, 66, 67, 69, 75, 77, 79, 81, 82, 83, 86, 87, 94, 97`

That cohort definition is necessarily post-hoc and gold-open: it came from the
already sealed r3 judgments. Once defined, however, the harness locks every row
by global ordinal, question ID, dated-question digest, and ordered population
digest. The successor cannot choose a more favorable subset, silently shift an
ordinal, or consult a reference answer while constructing evidence. The lock
and slicing implementation is in
[assay_hot_reduced30_construction.py](../../tools/assay_hot_reduced30_construction.py).

## Baseline reduced30 structural assay

The corrected baseline slice is sealed at
[corrected-r2/selection.json](../../eval_results/longmemeval-1m-hot-reduced30-construction-20260908-corrected-r2/selection.json)
with SHA-256
`4c0e5a97ea0a41184552d886f06c34d0f502f51dc6a24b0ce4c4b8d82633473c`.
It made zero provider calls.

| Metric | v6/r3 failure30 baseline |
|---|---:|
| Compiled facts | 1,468 |
| Rendered facts | 0 |
| Lane rejections | 7,224 |
| Mean context-token proxy | 9,883 |
| Maximum context-token proxy | 10,000 |
| Required slots represented by rendered episodes | 11/14 |
| Required slots matched by compiled facts | 9/14 |
| Required slots matched by rendered facts | 0/14 |

The result localizes the regression. The system could compile some useful
facts, but none reached the consumer, and the context ceiling was already
saturated. A fact compiler that only produces an audit object is not a recall
mechanism from the final LLM's perspective.

## r4 diagnostic: provider/audit cap coupling

The first v7 full100 diagnostic is sealed at
[r4/selection.json](../../eval_results/longmemeval-1m-hot-v7-spine-episode-fact-reserved-full100-20260908-r4/selection.json)
with SHA-256
`7e560a7973a533403ad20b0b085a743e66427d6def111c4a391464d8afa9351a`.

It preserved all 6,430 protected global raw rows, rendered every parent packet,
and omitted zero global raw evidence. Mean context-token proxy fell to
8,022.53, maximum context-token proxy was 8,817, and maximum workspace-token
proxy was 9,752. Despite that new headroom, it rendered **zero facts**.

The cause was cap coupling rather than evidence absence: the implementation
tested the large sealed fact/audit representation against the same space meant
for the compact provider representation. The audit object correctly retained
provenance, but its size prevented the much smaller `F` projection from being
admitted. This diagnostic is not an accuracy result and was not promoted.

## Repair sequence through r7

The r4-to-r7 repair kept v6/r3 immutable and changed only the additive
successor. The implemented composition rules are:

1. Specialized episode selection is independent of physical owner/transition
   expansion. Each lane gets its own cap, and the union reserves specialist
   selections before spending the physical remainder.
2. Transition breadth is typed and dynamic rather than universally consuming
   the maximum allowance.
3. Global raw parent evidence is monotone: the successor must preserve every
   selected parent row and records any omission as an invariant violation.
4. Candidate clauses are scanned across the active source domain. Compact facts
   are selected against a dedicated provider cap before optional episode
   expansion.
5. A selected candidate fact whose backing row was missed by episode selection
   triggers exact owner-envelope hydration. Selection happens before EM/raw
   deduplication; only the duplicate representation is removed afterward.
6. Provider text uses short `G`, `E`, and `F` labels. Full evidence IDs,
   collisions, mappings, and receipts stay in the sealed audit manifest instead
   of occupying the LLM context.
7. Typed reducers bind only to the final rendered fact slice and to backing raw
   evidence actually represented in the packet. Unsupported operations fail
   open and emit no advisory; set/count conclusions still require independently
   certified closure.

The successor packet builder is
[assay_hot_v7_spine_episode_fact_reserved_full100.py](../../tools/assay_hot_v7_spine_episode_fact_reserved_full100.py).
The reducer and provider-safe projection are
[hot_v6_typed_reducer.py](../../tools/matched_eval/hot_v6_typed_reducer.py)
and
[hot_v6_typed_reducer_advisory.py](../../tools/matched_eval/hot_v6_typed_reducer_advisory.py).

## Sealed r7 failure30 construction result

The historical r7 provider-free construction, current at that stage of the
repair, is sealed at
[r7/selection.json](../../eval_results/longmemeval-1m-hot-v7-spine-episode-fact-reserved-reduced30-20260908-r7/selection.json)
with SHA-256
`58dd33e8e2938afe36cae33615ffd1dd65b4b140060d75fcd56bcbee0d8119c8`.

| Metric | r7 failure30 |
|---|---:|
| Questions | 30 |
| Compiled facts | 226,673 |
| Rendered facts | 201 across 28/30 questions |
| Exact fact-backing hydrations | 38 |
| Unresolved selected fact backings | 0 |
| Parent packets preserved | 30/30 |
| Global raw evidence omitted | 0 |
| Mean context-token proxy | 8,706.733 |
| Maximum context-token proxy | 9,981 |
| Mean workspace-token proxy | 9,644.533 |
| Maximum workspace-token proxy | 10,918 |
| Required slots represented by rendered episodes | 14/14 |
| Required slots matched by compiled facts | 13/14 |
| Required slots matched by rendered facts | 12/14 |

Three final-slice typed advisories were admitted, with compact local support
labels and zero model calls:

- ordinal 15: `3 months`;
- ordinal 48: `the woman selling jam at the farmer's market`; and
- ordinal 97: `Yes`.

Ordinals 51 and 75 remain visible cap residuals in this historical r7 artifact.
Both report `mandatory_coverage_did_not_fit`: their compiled/audit ledgers
exist, but no fact slice or typed advisory was rendered. For ordinal 75, both
comparison sides exist in the active source domain, but r7's final packet
selected a restaurant-price anchor and the Tokyo lodging side rather than the
correct Maui lodging row. These are precise follow-up cases, not grounds to
claim that the whole packet is incomplete or that the final answer must fail.

## r9 structural repair and direct probes

The r9 work separates the two failures that r7 had made look alike.

Ordinal 51 (`41698283`) was primarily a fact-admission failure in r7, but it is
not an evidence-absence case. The relevant chronological raw chain survives as
`G69`, `G70`, and `G71`, including the specific 70–200 mm model and its later
recap. Under r9 the row reaches `selected_with_mandatory_coverage`: 8 facts are
selected, including all 3 mandatory facts; the fact-plus-advisory overlay is
493/500 tokens; final context is 9,926/10,000; final workspace is
10,859/11,000; exact backing hydration succeeds; and every parent raw row
survives. The remaining problem is semantic: the top-eight compact facts are
noise-dominated and do not express that the 70–200 mm lens is the latest state.
This is now a temporal ordering/coreference residual, not a packet-cap or raw
retrieval miss.

Ordinal 75 (`2318644b`) was a local-to-global source-linking and packing
failure. The correct Maui lodging evidence was already present in source
`2318644b::answer_eaa8e3ef_1`, but the r7 packet failed to bring the right
numeric neighbor forward. In the r9 direct probe, certified-lane source-local
numeric completion selects the relevant row stating `over $300 per night`
`a5bd78e0…`, hydrates its exact anchor and operand rows, maps them to compact
provider-local labels without binding a semantic slot, and renders the raw
evidence. Completion-aware backoff
reduces the fact slice from 7 to 6 while preserving all 4 mandatory facts. At
the successful mandatory admission stage, context/workspace are 8,482/9,420;
after optional episode backfill, the final provider packet is 9,891/10,829.
The combined provider overlay is 473/500 tokens and all parent rows survive.
The unresolved slot deliberately remains unresolved: `slot_bindings_added=0`,
`frontier_closed=false`, and the provider marks the material
`<SOURCE_LOCAL_CANDIDATES unbound>` rather than manufacturing a conclusion.

The r9 mechanisms responsible for those structural changes are:

1. Mandatory fact hydration starts from an empty episode set, then adds only a
   minimal lead/opener chunk plus each exact backing row. Its sealed policy is
   `empty_then_minimal_opener_plus_backing`. Optional episode neighborhoods are
   backfilled only after the mandatory fact packet fits.
2. Numeric completion is a bounded, source-local one-hop operation from a
   certified specialist-lane anchor. It may expose an exact numeric neighbor as
   raw evidence, but cannot add a slot binding or claim frontier closure.
3. Facts, typed advisory, and numeric completion share the existing 500-token
   provider-overlay budget. Optional facts back off in descending count while
   mandatory coverage is preserved. The successful ordinal-75 decisions are
   `provider_fact_count_backoff_for_numeric_completion` and
   `admitted_after_optional_fact_backoff`.
4. Compact provider labels remain separate from the exact-backing provenance
   manifest. The repair does **not** increase the 500-token cap and the direct
   probes make zero provider calls.

The first full locked reduced30 r9 construction ran for roughly four minutes,
then failed closed before sealing an artifact. Its exact-manifest final overlay
recalculation disagreed with the earlier admission accounting. Exhaustive
replay isolated one offender, ordinal 52: initial admission measured a filtered
manifest population, optional expansion changed the provider-label population,
and final rendering then exceeded the shared cap. The repair routes initial
admission, completion, fact backoff, every optional expansion, and final
rendering through one exact-manifest accounting function. Ordinal 52 now admits
10 facts, including all 4 mandatory facts, at 499/500 overlay tokens with final
context/workspace 7,773/8,701.

The repaired locked construction is sealed at
[r9/selection.json](../../eval_results/longmemeval-1m-hot-v7-spine-episode-fact-reserved-reduced30-20260908-r9/selection.json)
with SHA-256
`41d47ba7c0b42edbad0f408d1856ca563a49e11bdaf365a099ee283b9c8bfe28`.
It compiles 226,673 candidate facts, renders 226, and reaches
`selected_with_mandatory_coverage` on 30/30 rows. All 30 parent packets survive;
there are zero global omissions, unresolved fact backings, fallbacks, or
provider calls. Mean context-token proxy is 8,796.7, maximum is 9,926, maximum
workspace-token proxy is 10,859, and maximum shared overlay use is 499/500.
Rendered episodes cover 14/14 required slots; rendered facts cover 13/14.

## Proposed next additive layer: latest-state raw G-reference chain

Ordinal 51 suggests one further additive mechanism: for a question with an
explicit latest/current-state operator, emit a short chronological chain of the
already selected raw `G` references for the same entity or source. In the
concrete case, the provider would see the ordered `G69` → `G70` → `G71`
progression rather than relying on independently ranked facts to preserve the
transition. The chain should carry dates and compact local labels while its
exact IDs and receipts remain audit-only; it should expose ordering evidence,
not synthesize a new fact or claim closure.

This latest-state G-reference chain is a proposal only. It is **not implemented
in r9**, has no structural result, and has made no provider calls.

## Gold and provider boundary

The sealed r7 and r9 constructions and all construction probes were exactly
**zero-provider** and **gold-blind**. The dated question and authenticated
memory/cache/index artifacts were available. Reference answers were first
opened by the r9 judge preflight, after all 30 Terra predictions had been
sealed. Raw memory remains factual authority; compiled facts and typed
advisories are cited projections, not a replacement corpus.

This distinction matters because the cohort identity itself came from a prior
gold-open analysis. A locked residual benchmark is legitimate for repair
development, but its eventual score is a targeted regression result, not an
unseen generalization estimate.

## Performance implication and ingest-side plan

Profiling found roughly **7,016 compiled candidate facts per question to emit
at most 20**. That fan-out is unnecessary query-time work even though the final
packet is bounded. It argues for an ingest-time, per-source clause sidecar:
extract and authenticate compact clauses as each conversational source closes,
store their exact raw evidence coordinates, and let query time rank bounded
source-local clause lists before hydrating raw text.

That sidecar is a plan, **not an implemented result in r7 or r9**. No speedup,
recall guarantee, or ingest-throughput number is claimed for it here.

## Sealed r9 reduced30 accuracy

The complete lifecycle is sealed under
[the r9 Terra/Sol evaluation root](../../eval_results/longmemeval-1m-hot-v7-spine-episode-fact-reserved-reduced30-terra-sol-20260908-r9/).

- answer preflight SHA-256:
  `907085d1c923f12358d4c942652e30b8ed6e969c2cde42d59d1b1d232623e5d7`;
- Terra answers SHA-256:
  `35a69ff3c999ea29fbcc23b234512c539c487c79e38215e577ef195ca824bf3f`;
- judge preflight SHA-256:
  `74a881d5a01c5e20448a6a8ebeb6ece8a403981edcb68b2b8c6df361cd899725`;
- Sol judgments SHA-256:
  `799908614ab4c3f18fed75bec7db5d2be75738893b82b56621caa86c4e92cb72`;
- Terra: exactly 30 new calls, zero retries, zero initial checkpoint hits,
  53.203 seconds batch wall time; and
- Sol: exactly 30 new calls, zero retries, zero initial checkpoint hits,
  47.673 seconds batch wall time.

Both phases replay from 30 authenticated checkpoints with authorization zero.
The call, retry, checkpoint-hit, and wall-time figures are observations from
the live runner output; unlike the four artifact hashes, they are not
independently sealed run-receipt files.
The valid result is **11/30**, with recovered global ordinals
`6, 15, 17, 25, 40, 43, 48, 59, 66, 86, 87`. The remaining 19 are
`5, 14, 36, 42, 49, 51, 52, 53, 61, 67, 69, 75, 77, 79, 81, 82, 83, 94, 97`.

| Category | Correct | Total |
|---|---:|---:|
| Temporal reasoning | 7 | 9 |
| Multi-session | 3 | 11 |
| Single-session user | 1 | 4 |
| Single-session preference | 0 | 5 |
| Knowledge update | 0 | 1 |

Judge preflight initially failed closed before any Sol call because the runner
used the historical ten-question *development* concatenation loader while the
lock addresses the ten-shard, 100-question *validation* population. The local
dataset and split hashes were correct. The runner now reconstructs and pins the
validation population SHA-256
`9b8ad9337cfece1306358d0e03682a977f1b289a14b6ff7bfe40c90e6e2cb246`,
flattens its ten shards, and verifies every global ordinal, question ID, and
dated-question digest. Its focused lifecycle suite passes 9/9.
The final combined implementation, reducer/advisory, lifecycle, and evaluator
profile suite passes 107/107 after that repair.

Because failure30 contains exactly the rows scored wrong in v6/r3, 11/30 means
11 targeted recoveries. If all 70 formerly correct rows remained correct, the
arithmetic total would be 81/100. That is a conditional projection, **not** a
measured r9 full100 score; r9 could still regress former successes.

## Current decision

Do not start the full100 r9 run: 11/30 is below the 25/30 residual gate needed
even conditionally to reach 95/100. Preserve r9 as the packet-conservation
result and split the remaining 19 by specialist need. The immediate additive
experiments are: a latest/current-state raw-reference chain; a closed-frontier,
semantically deduplicated event-count path; preference-specific episodic
summaries; deterministic use of a supported typed advisory; and corrected
calendar arithmetic. Test each mechanism on its eligible subset before
recombining them. The operational continuation is recorded in
[Research Log 121](121%20-%202026-09-08%20-%20r9%20reduced30%20execution%20handoff.md).
