# Method eligibility, failure attribution, and apparatus cleanup

**Date:** 2026-09-01

**Status:** evaluation-contract analysis plus an initial behavior-preserving
performance refactor; the common outcome ledger remains proposed

## Decision

The next bounded engineering tranche should clean up the failure-attribution
and retrieval-apparatus path before another retrieval mechanism is added.

This is not a proposal to flatten the memory architecture or replace its
specialists with one generic retriever. The mechanisms should retain their
separate scopes, budgets, and evidence. The cleanup target is the repeated
machinery around them: applicability, execution, candidate lifecycle, frontier
state, post-selection deduplication, binding, operator consumption, answer
fallback, and posthoc attribution.

The working hypothesis is:

1. a common outcome projection will make method-local failures distinguishable
   from routing, packing, terminal, corpus, and reference failures;
2. removing repeated joins, projections, token counts, and population walks may
   reduce provider-free construction and replay time; and
3. recovered latency can then be spent on a wider or more completely resolved
   search frontier at the same operational budget.

Only the first statement follows directly from the present code. The second
and third are performance and recall hypotheses that require measurement.
Cleanup alone is not an accuracy result.

## The distinction we need

An incorrect final answer is not a useful method diagnosis. Before assigning a
failure, the evaluator must answer these questions in order:

1. Did the sealed question-only contract declare the method applicable?
2. If applicable, did the method actually run?
3. Which authenticated population did it search, and was that frontier closed?
4. Did an assigned target reach candidate generation, selection, deduplication,
   packing, provenance binding, operator consumption, and the answer prompt?
5. If the evidence survived, did fact compilation, deterministic execution,
   answer synthesis, validation, or the benchmark reference fail?

Applicability must be fixed before execution outcome is known. An eligible
method cannot turn its own miss into `out_of_scope`. Likewise, an inapplicable
runtime route is not automatically a correct out-of-scope decision: the
postseal target-owner join may show that the router failed to activate the
method that owned the target.

The resulting evaluation labels are:

| Label | Required evidence |
| --- | --- |
| `OUT_OF_SCOPE` | The sealed ownership contract assigns the method no target obligation, and the question-only applicability decision is consistent with that assignment. |
| `ROUTING_FAILURE` | The method owns a target obligation, but the question-only route marks it inapplicable or never activates it. |
| `UNRESOLVED_FRONTIER` | The method owns an obligation, ran, and did not reach the target, but eligible work remains unsearched or unresolved. This is an end-to-end miss, not yet proof that the method's search rule failed. |
| `METHOD_FAILURE` | The method owns an obligation and the authenticated lifecycle identifies the first loss inside its closed or actually searched scope: discovery, ranking, selection, admission, packing, binding, linking, or consumption. |
| `DOWNSTREAM_FAILURE` | The required evidence is provider-visible or operator-consumed, but fact compilation, execution, answer synthesis, validation, or judging fails. |
| `DATA_OR_REFERENCE_BOUNDARY` | A separate authenticated audit proves the required evidence is absent, contradictory, or inconsistent with the accepted reference. An open search cannot establish this label. |
| `SATISFIED` | The method's owned obligation reaches its declared terminal boundary. Answer correctness remains a separate measurement unless answering belongs to that method's contract. |

These labels are target-aware evaluation conclusions. The production path must
remain gold-blind.

## What is already present

The code already records most of the facts needed for this distinction.

| Existing seam | What it establishes | Remaining limitation |
| --- | --- | --- |
| [`StageTrace`](../../tools/matched_eval/contracts.py#L295-L361) | Candidate, selected-before-dedup, duplicate, not-admitted, and admitted IDs; token and provider counts; canonical `ADDED`, `NO_OP`, `OVERFLOW`, `INVALID`, and `FAILED` dispositions. | The stage-level reason is free text, and an empty candidate set alone does not distinguish inapplicability from a miss. |
| [`SemanticResidualEligibilitySignals`](../../tools/matched_eval/semantic_residual_eligibility.py#L165-L265) and its decision | Specialist applicability, route gap, bounded/open frontier, unresolved slots, sufficiency, abstention, and reconciliation state. | It is a downstream fallback gate, not a universal ownership or causal-failure contract. |
| [`EvidenceFrontierReceipt`](../../tools/matched_eval/typed_operator_adapter.py#L416-L480) | Exhaustive, bounded, or open upstream evidence; represented, omitted, rejected, and unresolved handles. | Its available population is only the evidence exposed upstream, not necessarily the complete immutable store. |
| [`GlobalTreeFrontier`](../../tools/matched_eval/semantic_global_completion.py#L753-L841) and [`ClassifiedResidualFrontierReceipt`](../../tools/matched_eval/semantic_residual_search.py#L2216-L2328) | Retained, definitely negative, packed, protected-duplicate, and unresolved leaf or segment partitions, with explicit closure. | Completeness is relative to the authenticated semantic index and classifier contract; it is not unrestricted semantic-absence proof. |
| [`LocalReinjectionFrontier`](../../tools/matched_eval/source_group_reinjection.py#L759-L875) | Source-local packed, duplicate, and budget-unpacked segments; unresolved obligations; whether global search is still needed. | It deliberately cannot claim global support closure. |
| [`ObligationCoverageReceipt`](../../tools/matched_eval/source_gate_controller.py#L874-L922) and [`GateStopReason`](../../tools/matched_eval/source_gate_controller.py#L1014-L1055) | Unresolved, partial, conflicted, or satisfied obligations; pending physical work; closed frontier; candidate exhaustion, caps, or no-progress stops. | Exhaustiveness is declared by the caller's eligible-frontier receipt, so the receipt must stay bound to its exact population basis. |
| [`build_typed_connectivity_ledger`](../../tools/matched_eval/typed_connectivity_ledger.py#L94-L279) | First disconnection after accepted evidence: provenance, source group, role, time, discourse, allowed handle, validation, or operator consumption. | It cannot observe a method that never ran or evidence that never became an accepted item. |
| [`_failure_transition`](../../tools/analyze_locked_typed_memory_final_targets.py#L1276-L1307) | Posthoc transitions such as missing at retrieval, lost at dedup/lane/merge/fit/binding, missing relation, or not consumed by the operator. | It is a bespoke gold-bearing analysis over a bounded population, not a shared runtime/evaluation contract. |
| [Locked target-owner plan](../10%20-%20Research%20Log/data/longmemeval-locked-100-target-owner-plan-v1.json) | Each of 263 declared targets has exactly one primary owner, enabling per-method denominators after the runtime artifacts are sealed. | It is explicitly posthoc and runtime-forbidden. It cannot be used for production routing or selection. |

There are also narrower vocabularies such as `no_applicable_specialist`,
`specialist_proofless`, `question_only_route_ineligible`, and assay-specific
loss stages. They are useful locally, but they do not form one complete
question-by-method matrix.

The selected failure/frontier surface inspected for this note spans 17,820
lines across eleven modules. Within that bounded surface there are 205 calls to
`projection`, 91 calls to `identity_sha256`, and 28 calls to `count_tokens`.
Those static counts show duplicated apparatus pressure and review surface; they
do **not** prove where runtime is spent.

## Missing common contract

The smallest useful addition is a provider-free
`QuestionMethodOutcomeReceiptV1`. It should be a read-only projection over
existing sealed artifacts, not a new retrieval authority and not a rewrite of
historical receipts.

There must be exactly one row for every sealed question and every declared
method. Inapplicable and not-run rows must remain visible so failures cannot
disappear from a denominator.

The receipt should keep separate axes rather than collapse the lifecycle into
one lossy enum:

| Axis | Canonical states |
| --- | --- |
| Applicability | `applicable`, `not_applicable` |
| Execution | `not_run`, `complete`, `failed_precondition`, `missing_external_result`, `invalid_external_result` |
| Discovery | `not_run`, `zero_candidates`, `candidates` |
| Selection | `not_run`, `zero_selected`, `selected_all`, `selected_with_budget_skips` |
| Deduplication | `not_run`, `all_retained`, `authority_transferred`, `mixed_retained_and_transferred` |
| Interpretation | `not_run`, `definitely_irrelevant`, `unresolved`, `facts`, `mixed` |
| Closure | `not_required`, `open_missing`, `open_unresolved`, `closed_within_declared_population` |
| Answer | `not_run`, `passthrough`, `invalid_fallback`, `kept_parent`, `changed` |

Each row should bind:

- population, question, dated-question, method, route, and upstream receipt
  identities;
- candidate, selected, skipped, post-dedup, final-handle, fact, typed-link,
  and unresolved populations;
- evidence and prompt-token budgets, output reserve, logical and physical
  calls, and retained transformer-token state;
- the exact existing child receipts from which every state was projected; and
- a deterministic receipt over canonical fields only.

The following invariants are mandatory:

1. `not_applicable` implies `not_run`, zero physical calls, and passthrough or
   no answer contribution. A runtime failure cannot alter applicability.
2. Applicable `zero_candidates`, `zero_selected`, open closure, overflow,
   invalid output, and missing external output remain visible failure states.
3. Candidates partition exactly into selected and skipped; selected evidence
   partitions exactly into retained and authority-transferred duplicates.
4. Every provider-visible handle maps to one authenticated method contribution.
5. Every selected leaf has one interpretation outcome, and every required
   obligation records covered, missing, conflicted, or unresolved state.
6. Replay reproduces the deterministic outcome receipt byte-for-byte.
7. The receipt is gold-blind. Target ownership and judge verdicts are joined
   only in a separately sealed posthoc diagnostic extension.

The posthoc extension should combine owned targets, witnessed lifecycle stages,
and answer verdicts to assign the evaluation labels in the first table. This
keeps a correct `OUT_OF_SCOPE` judgment separate from a router that merely
claimed not to apply.

## Cleanup strategy

The first implementation should add one normalized analysis adapter and leave
every production mechanism untouched. Existing artifacts are its input; a
common ledger and replay are its output.

Once that projection reproduces the current bespoke audits, cleanup can proceed
from the outside inward:

1. **Normalize state and reason vocabularies.** Centralize enums and adapters
   while retaining backward readers for existing sealed artifacts.
2. **Consolidate repeated joins.** Build question, method, candidate, handle,
   leaf, and receipt maps once per authenticated population instead of
   reconstructing them independently in each audit.
3. **Consolidate deterministic projections.** Reuse canonical serialized bytes,
   hashes, and token counts inside one immutable run when identity and tokenizer
   bindings match.
4. **Migrate analyses one at a time.** Compare the common ledger against the
   independent-closure, residual, specialist, semantic-global, connectivity,
   and typed-final reports before deleting any bespoke path.
5. **Keep mechanism internals separate.** Numeric, temporal, profile, episodic,
   linker, local reinjection, and global search behavior should not be merged
   merely because their outcome projections share a schema.

This is structural cleanup, not code golf. Shorter code is useful only when it
removes duplicate semantics and preserves readable method boundaries.

### Concrete first-pass hotspots

A read-only code audit found several behavior-preserving candidates that are
specific enough to benchmark before any architectural rewrite:

| Priority | Current repeated work | Safe first treatment |
| --- | --- | --- |
| 1 | [`_SealedRecord.receipt_sha256`](../../tools/matched_eval/source_gate_controller.py#L108-L119) rebuilds the complete body and SHA on every property access; `projection()` immediately requests it again. | Cache the receipt inside the immutable record without changing its projected fields or canonical bytes. |
| 2 | [`SemanticResidualIndex.cell_by_id` and `manifest_by_node_receipt`](../../tools/matched_eval/semantic_residual_search.py#L1359-L1367) rebuild maps on every access. Source-local and semantic-global helpers separately rebuild histories, cell inventories, span maps, and segment maps. | Build one immutable query-local index inventory and pass or cache it for the lifetime of the authenticated index. |
| 3 | The residual constructor calls the validator after [`semantic_binary_search`](../../tools/matched_eval/semantic_residual_search.py#L2975-L2989), even though the search validates before returning; after-union selection then immediately invokes full replay after constructing the same result ([`build_after_union_selection`](../../tools/matched_eval/after_union_fact_closure.py#L502-L519)). | Keep replay as an explicit audit API, but remove duplicate validation/replay from measured construction paths after parity tests prove the boundary. |
| 4 | Local, residual, and global skip-and-continue packers repeatedly serialize and tokenize the entire growing payload. The typed salvage loop rebuilds frontier, handles, JSON, and token count for each trial and eviction ([`_packet_with_budget_salvage`](../../tools/matched_eval/typed_operator_adapter.py#L1249-L1345)). | Cache identical fit probes and immutable row projections, share one exact pack kernel, and retain a final full-payload token verification. Do not replace exact tokenization with naïve per-row addition because BPE boundaries are not additive. |
| 5 | Global best-first priority computes node lane bounds while pushing and again after popping ([`_priority_key` and `_search_tree_best_first`](../../tools/matched_eval/semantic_global_completion.py#L1087-L1185)); semantic tree node enumeration and descendant token sums are also recomputed on property access ([`SemanticSearchTree`](../../tools/matched_eval/semantic_binary_search.py#L205-L304)). | Cache query-local node bounds, preorder nodes, and immutable token counts; preserve ordering and all projected values. |
| 6 | Source-gate obligation assessment normalizes the same fact text inside obligation-by-fact loops, while plan candidate lookup and replay repeatedly linearly scan or revalidate prefixes. Typed operator preflight similarly reconstructs usable-item and semantic-key views. | Precompute normalized fact/operator views and query-local lane/ID maps; validate a replay lifecycle once, then advance through an internal already-validated path. |

The safest initial sequence is receipt caching, immutable inventory reuse, and
redundant-validation removal; then query-feature caches and operator/coverage
views; then exact packer consolidation. Shared low-level receipt helpers and
tree-storage changes should wait until golden receipt and payload tests cover
the smaller changes.

## Initial cleanup checkpoint — 2026-09-01

The first bounded cleanup tranche is now implemented. It deliberately changes
neither retrieval policy nor any evidence, prompt, receipt, or score. Its
purpose is to remove repeated deterministic work and use the resulting profile
to decide where further cleanup belongs.

The fixed six-file provider-free benchmark covered source gating, semantic
tree search, typed operator adaptation, residual search, global completion,
and source-group reinjection. Before the change it ran 73 tests in 60.19
seconds. The same file set after the final reviewed change ran 74 tests in
23.71 seconds; the additional test is the new manifest-materialization
regression. This is 36.48 seconds, or 60.6%, less wall time despite the extra
test. It is a local test
slice rather than a production-throughput claim.

The largest measured defect was concrete: the residual classifier requested
`manifest_by_node_receipt` at every visited tree node, and that property rebuilt
the complete receipt-to-manifest dictionary on every access. The refactor
materializes it once per classifier, precomputes immutable query/manifest/cell
term sets, and computes each query-term document frequency once. Representative
residual tests changed as follows:

| Provider-free case | Before | After |
| --- | ---: | ---: |
| q42-like absence/partition closure | 5.39 s | 0.13 s |
| stored-chunk centroid path | 3.31 s | 0.14 s |
| missing-vector fail-open path | 3.29 s | 0.25 s |
| q74-like residual path | 2.10 s | 0.15 s |
| q65-like residual path | 1.67 s | 0.14 s |
| IDF specificity path | 1.53 s | about 0.14 s |

The rest of the tranche makes the same kind of bounded change:

- global best-first search materializes its manifest map once and reuses each
  node's lane bounds between heap priority and visit recording;
- immutable semantic tree nodes cache descendant token counts, while the tree
  caches preorder traversal and total tokens outside its projected identity;
- source-gate projection builds its body once and seals that exact body, while
  deliberately resealing later accesses so forced mutation remains evident;
- typed-packet salvage memoizes identical exploratory fit probes but still
  performs an uncached final render and exact hard-cap token check; and
- retained/pruned membership sets are built once instead of inside repeated
  comprehensions.

Golden projection, receipt, payload, constructor, ordering, rejection, and
tamper-validation tests cover the new caches. Semantic-tree cache slots are
excluded from dataclass fields, constructors, equality, representation, and
sealed projections; copy and pickle reconstruction repopulate and revalidate
them. The authoritative final payload check is intentionally not cached.

The wider regression pass also exposed a pre-existing prompt-version boundary:
the reduced-specialist loader always selected the legacy typed-answer renderer,
while four sealed v4 rows bound the newer resource-preserving renderer. Their
stored message hashes matched the newer version exactly and their prompt token
counts were consistently 19 tokens above the legacy reconstruction. The repair
does not infer a version from date or artifact name. It tries only the two
explicitly frozen renderers and accepts the unique one whose message SHA and
exact token count match the sealed terminal; otherwise it fails closed.

Profiling also prevented a false optimization target. One global-completion
test took 2.727 seconds, but its fixture `_build` accounted for 2.571 seconds;
the actual global search accounted for about 0.137 seconds. The fixture made
423 SQLite commits, accounting for about 1.791 seconds in that profile. These
figures identify store/test construction batching as the next apparatus seam;
they do not justify changing global-search semantics.

This checkpoint narrows the next work:

1. keep immutable cell, manifest, and normalized feature inventories at the
   shared-index lifetime when many prompt ticks reuse one memory store;
2. batch authenticated store writes in builders and test fixtures without
   weakening transaction or receipt boundaries;
3. precompute source-gate candidate and normalized fact/obligation views before
   changing the gate algorithm;
4. profile source-local history hydration and the remaining exact packers before
   extracting a shared kernel; and
5. build the common question-by-method outcome ledger so saved work can be
   allocated specifically to applicable, unresolved frontiers rather than to
   every method indiscriminately.

The important result is diagnostic as well as operational: a real inner-loop
apparatus defect was removed, whereas the remaining visible global-test cost is
mostly construction I/O. No new retrieval or answer score follows from either
finding.

## Performance work inside the cleanup

Performance data must live in a separate
`QuestionMethodPerformanceSidecar`, keyed by the deterministic outcome receipt.
Wall time, cache state, and machine load must not alter replay identity. This
continues the identity boundary established in
[Research Log 34](../10%20-%20Research%20Log/34%20-%202026-08-22%20-%20Cumulative%20apparatus%20performance%20diagnosis.md#receipt-and-identity-boundary-for-v2).

The sidecar should record, per method and phase:

- route, discovery, selection, deduplication, linking, leaf classification,
  fact compilation, packing, answer, and judge elapsed time;
- rows, sources, partitions, cells, nodes, candidates, and bytes visited;
- vector comparisons, database calls and returned rows, cache/checkpoint hits,
  canonical serializations, hashes, and tokenizer calls;
- evidence tokens, complete prompt envelope, logical calls, physical calls,
  provider input/output/cache tokens, and peak resident memory; and
- cold and warm p50, p95, and maximum latency.

The first profiler should test these hypotheses rather than assume them:

1. repeated canonical projection, JSON serialization, hashing, and tokenization
   consume material time in the large immutable receipt graph;
2. repeated list/set construction and cross-artifact joins rescan the same
   candidate populations;
3. namespace, source, partition, and normalized-query features are rebuilt
   across method-local analyses that could safely share authenticated caches;
4. evidence is hydrated or copied earlier and more often than selection needs;
   and
5. semantic scoring contains scalar Python loops that would benefit from exact
   batched NumPy operations.

JAX or Numba should not be introduced until profiling shows a stable numeric
kernel dominates. Most likely wins here are fewer population passes, cached
immutable projections, batched database access, vectorized scoring, and lazy
evidence hydration. A compiler or receipt-identity change requires a new
version even if final evidence happens to match.

There is precedent for structural performance work paying off. The original
apparatus audit measured 12,891,681 source-row hash visits in one completed
store, while the compact resumable namespace successor later reduced a
50-minute import to about 13 minutes and replayed in about 20 seconds. Neither
result proves the present bottleneck, but both justify measuring structural
work before adding hardware or another model call.

## Execution roadmap

### Phase 0 — freeze the baseline

- Record current artifact hashes, replay outputs, wall time, peak memory, and
  stage counters.
- Use deterministic fixtures for every applicability and lifecycle state.
- Use the exact-11 provider-free workload only as a profiling diagnostic, not a
  deployable score claim.

### Phase 1 — build the read-only outcome ledger

- Emit the complete question-by-method matrix from existing receipts.
- Replay it byte-identically.
- Add the postseal owner/witness join as a separate command and artifact.
- Reproduce existing failure-transition and connectivity counts exactly.

### Phase 2 — profile before refactoring

- Measure cold and warm construction and replay.
- Attribute population walks, serialization, hashing, tokenization, database,
  vector, and evidence-hydration costs.
- Separate experiment certification cost from production query-path cost.

### Phase 3 — behavior-preserving cleanup

- Refactor only the highest measured repeated work.
- Require identical candidate order, evidence, authority transfer, handles,
  facts, closure, prompts, semantic receipts, and replay bytes.
- Keep old artifact readers and delete a bespoke path only after parity tests.

### Phase 4 — convert speed into recall capacity

- Hold wall time, evidence-token budgets, and model-call budgets fixed.
- Spend recovered compute only on eligible unresolved frontiers: more source
  groups, residual cells, semantic branches, or closure checks.
- Compare per-method conditional target reach and the union of method-owned
  targets. Do not increase every method's budget indiscriminately.

### Phase 5 — quality gate

- Run the provider-free locked full100 construction and replay first.
- Report candidate, selected, admitted, packed, bound, consumed, and closed
  rates per applicable method, plus union target coverage and protected-parent
  regressions.
- Authorize answer and judge calls only after the structural gate passes.
- A full100 answer score and an independent confirmation set remain necessary
  for a 95% claim.

## Acceptance criteria

The cleanup tranche succeeds if it produces all of the following:

1. one auditable outcome row for every question and declared method;
2. no eligible miss mislabeled as out of scope;
3. exact first-failure localization from routing through terminal consumption;
4. byte-identical behavior for the first refactor tranche;
5. measured provider-free wall-time, memory, and work-count improvement, or a
   clear result showing that the hypothesized apparatus costs are not material;
   and
6. a same-budget experiment showing whether the saved work closes more owned
   target obligations without regressing the protected parent.

If cleanup improves attribution but not runtime, it is still useful, but it has
not earned a recall claim. If it improves runtime but the additional frontier
does not recover owned targets, then performance was not the remaining recall
bottleneck. If evidence is already consumed and answers remain wrong, work must
move to typed operators, scoped validation, and terminal synthesis rather than
more retrieval.

## Claim boundary

This note records an architectural decision, a measured first cleanup tranche,
and the remaining testable hypotheses. It does not implement the common
question-by-method ledger, broaden any retrieval frontier, or change an answer
score. Existing sealed artifacts and their historical terminology remain
authoritative for their campaigns.
