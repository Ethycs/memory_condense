# From top-k recall to proof-carrying factual retrieval

**Status**: IMPLEMENTED for the locked v3 treatment; development evidence is
strong, held-out answer accuracy remains unmeasured
**Date**: 2026-08-18
**Applies to**: factual lookup, derived scalar, fixed-cardinality,
enumeration, chronology, and current-state questions over long conversations

## Claim

The factual retrieval problem was not fixed by one better similarity score.
It was fixed by separating five different correctness obligations that top-k
retrieval had conflated:

```text
reachability -> identity -> sufficiency -> packet integrity -> proof scope
```

1. **Reachability:** can an answer-bearing source enter the candidate set?
2. **Identity:** which mentions are the same event, and which are distinct?
3. **Sufficiency:** did the packet retain every operand or list member needed
   to compute the answer?
4. **Packet integrity:** are revisions, roles, dates, and conflicts represented
   without a plausible distractor replacing the answer?
5. **Proof scope:** what corpus or partition did the system actually inspect,
   and may it honestly close the tail?

The resulting architecture is **proof-carrying retrieval**. The answerer still
receives ordinary source text, but selection carries a machine-checkable
receipt explaining candidate reachability, structural hypotheses, scope,
reservations, conflicts, closure, and exact cost under the frozen local token
proxy. A high model score can rank evidence; it cannot manufacture a
completeness proof. When provider input usage is available, that nonzero
provider measurement remains authoritative.

## Why the original system missed facts

### Similarity pools are not recall proofs

Dense and BM25 retrieval can rank an exact fact highly once its source is in
the pool. They cannot prove that every source needed for an enumeration entered
the globally truncated pool. Source diversity and hierarchy operate on the
sources they can see; they do not recover an invisible source.

This distinction became concrete on the six-museum question. The required
source was reachable only after scanning all content chunks in the routed
partitions. The scan also found a seventh plausible visit that offline gold
joining later identified as non-gold. Gold IDs never enter retrieval. That was
useful evidence: the problem was not “retrieve six rows,” but “identify six
canonical completed visits while labeling a competing retrospective mention.”

### Set coverage can damage scalar questions

Questions such as “how many days apart,” “how many weeks,” or “how many pages
remain” require two or three operands, not an exhaustive set of answer
entities. Compiling them as `COUNT` caused the selector to reserve many
irrelevant clusters. Exact operands were sometimes already selected but fell
behind a prefix or packet boundary.

Likewise, “How many followers do I have now?” is a current scalar lookup. The
old set program inspected 83 candidates, formed 51 clusters, and reserved 17
representatives even though the two state updates were already in the baseline
packet. More set reasoning created more opportunities to lose the answer.

### A mention is not an event

The five-concert question contained eight direct-looking mentions. Some were
artist-detail restatements or cross-source recaps of the same concert; other
concerts in one source were genuinely different. Deduplicating by source loses
events. Treating every mention as distinct inflates cardinality and consumes
the prompt. The useful identity is transient and query-relative: completed
event type plus venue or location when those fields are unambiguous.

### Roles and temporal recaps can be credible but wrong

A forced-choice scorer preferred polished assistant recommendations over short
user statements for one fixed-three query. The text looked answer-like, but it
had the wrong evidence role.

Another query contained an explicit “started today” boundary, a later endpoint,
and an approximate recap saying “about six weeks.” Packing all three leaves a
responder with a contradiction even when the exact duration is derivable. A
generic similarity merge can also bury the original onset behind the later
recap.

### More evidence can lower answer accuracy

For the museum question, all six exact gold chunks reached the packet. A
seventh plausible gallery visit also reached it, and the answerer substituted
that distractor for one required museum. Source recall was perfect while answer
accuracy was wrong. Context is not monotonically useful: after sufficiency,
unsupported alternatives can reduce reliability.

## The repair

### 1. Keep several recall routes, but treat them as hypotheses

The coarse route combines BGE-M3, BM25, role weighting, source TF-ISF, source
hierarchy, partition routing, local-neighbor search, and bounded causal
associations. These routes maximize ways for a direct fact to become visible.
They are additive and fail open: graph or model candidates do not erase the
last direct anchors merely because a learned route assigns a stronger score.

The causal graph separately repaired outcome reachability. Completed
prompt/response episodes are covered by bounded slices, compact IDs and scalar
edges are retained, and later queries may traverse two bounded hops. The
[causal replay](../10%20-%20Research%20Log/09%20-%202026-08-16%20-%20Causal%20binding%20reaches%2097.4%20percent%20evidence%20recall.md)
progressed from 35/39 original literal hits to 36/39 with packing, 37/39 with
rank consolidation, and 38/39 with Qwen-weighted consolidation, under the same
1,600-token evidence cap.

### 2. Compile the question before invoking set coverage

The query compiler now distinguishes:

- `SINGLE` lookup and derived scalar questions;
- narrow first-person current-possession scalar questions;
- `FIXED(k)` enumerations;
- `ALL` enumerations;
- `COUNT` of actual event/entity sets; and
- ordering and as-of requirements.

Derived scalars and current-state possession bypass the expensive set selector.
Their operands flow through the ordinary query-aware information-gain packer,
which restored the exact dates, onset/endpoint, page total/current page, and
state updates in offline reconstruction. The rule is intentionally narrow:
ordinary event counts and historical counts remain set queries.

### 3. Scan typed evidence where top-k cannot establish reachability

For supported fixed/all event shapes, the condenser exhaustively scans the
content rows in the selected active partitions, then reduces them to a bounded
admitted frontier. It records:

- total and inspected chunks and sources;
- structural rows and reduced hypotheses;
- candidates already present, admitted, replaced, or truncated;
- alternatives, ambiguity, overflow, wrong-role and out-of-time rejections;
- the scan contract and exact source/partition scope; and
- a snapshot of required `(chunk_id, route)` pairs.

This scan is structural admission, not a model answer. It guarantees that a
typed candidate inside the audited scope can reach the selector. It does not
claim the approximate top-k partitions equal the corpus.

Snapshot validation runs again immediately before packing. A changed chunk
high-water mark, removed/changed active row, or newly injected active route
invalidates the proof. Ordinary non-active rows may be reranked without
destroying the audited subset contract.

### 4. Reduce mentions to conservative event identities

Performance queries use a transient `performance_event_key` only when one
completed episode sentence yields an unambiguous type plus venue/location.
Equal keys contract across details and cross-source recaps. Different keys
remain distinct even inside one source. Keyless or ambiguous rows remain
alternatives and make completeness fail open.

On the locked concert question this reduced eight direct structural rows to
five hypotheses while retaining all five answer-bearing primaries. The rule is
not a museum/concert benchmark dictionary; it is a conservative grammatical
event identity with explicit abstention tests.

### 5. Align reservation with the required evidence role

The query program separates a soft preferred role from a high-confidence
required role. Explicit retrospective language can require assistant evidence;
first-person completed actions can require user evidence; ambiguous questions
remain unconstrained.

For non-typed fixed-k queries, reservation tiers are:

1. credible clusters containing required-role evidence;
2. stable non-null required-role representatives
   (`role_aligned_fixed_frontier`); then
3. cross-role credible evidence as a fail-open tail.

Opposite-role rows are never deleted. They simply cannot consume the only
reserved fixed-k slots when the query itself proves which speaker owns the
facts. The forced-choice prompt also renders candidate author role so
first-person pronouns are interpreted relative to the correct speaker.

### 6. Preserve exact temporal boundaries and suppress only proven conflicts

Duration queries prefer a unique explicit onset boundary plus endpoint over a
later approximate recap. Suppression is allowed only when the recap is
provably conflicting and approximate; otherwise both rows remain. The decision
is recorded by chunk ID and basis, while source text remains outside the scalar
report.

This is deliberately narrower than “newest statement wins.” Newness alone
cannot tell whether a later sentence is a correction, estimate, plan, or recap.

### 7. Close the tail only with an honest scope contract

Fixed-k post-coverage closure can remove distractors after exactly `k`
structural identities are proven. The globally or authoritatively complete
path requires:

- an exhaustive typed scan;
- exactly `k` structural hypotheses and reserved representatives, with no
  overflow or truncation (raw structural rows may be greater than `k` because
  duplicate mentions can contract to one hypothesis);
- valid scan contract and source counts;
- partition inventory and selected-scope provenance;
- globally or authoritatively complete scope; and
- a current validated snapshot.

Approximate selected partitions are fail open by default. The development
museum policy uses the only other permitted path: an explicit frozen
selected-scope opt-in. It reports:

```text
closure_scope = selected_scope_policy
closure_global_recall_guaranteed = false
```

That label is not cosmetic. It prevents a successful selected-scope
experiment from becoming a false corpus-completeness claim.

### 8. Make the packet and cache reproducible evidence

The packet path counts exact rendered text under the frozen tokenizer proxy,
keeps direct evidence fail open, and never persists request-derived K/V,
attention, residual, or activation state.

Validation caches are content-addressed and read-only. Receipts bind the exact
sample, SQLite/ANN bytes, BGE revision/checkpoint/execution identity, compiled
to causal cache link, implementation hash, environment hash, split, policy,
and prompt proxy. A scored validation run cannot build a missing cache while
held-out questions are live.

These controls do not improve semantic recall directly. They prevent a result
from being credited to different code, mutable state, or an accidentally
rebuilt corpus.

## The resulting pipeline

```text
question
  -> typed query program
  -> multi-route direct candidates
  -> bounded typed scan inside explicit scope
  -> conservative event/role/time reduction
  -> optional generation-free Qwen grouping
  -> representative-first reservations
  -> conflict handling
  -> exact hard-budget packing
  -> scope-qualified closure receipt
  -> answerer sees source text only
```

The Qwen components are bounded inspectors, not memory stores. The two-layer
Qwen3-8B prefix supplies QK/OV affinity without an LM head; the Qwen3-0.6B
choice model performs forced-choice conditional-likelihood A/B scoring (and
supports bounded choice sequences) with K/V cache disabled. Only text-free
scalar reports and durable source IDs cross a call boundary.

## Evidence

The [final v3 development replay](../10%20-%20Research%20Log/16%20-%202026-08-18%20-%20V3%20retrieval%20freeze%20and%20validation%20campaign.md)
used a 1,039,203-token-proxy, 5,400-turn composed conversation:

| Metric | Result |
| --- | ---: |
| Raw evidence-source coverage | 100% |
| Packed evidence-source coverage | 100% |
| Questions with every packed evidence source | 10/10 |
| Scored answer-value components | 11/11 |
| Mean / maximum returned context | 1,985.6 / 2,219 tokens |
| Selector / score-provider fallbacks | 0 / 0 |
| Maximum retained request-token state | 0 bytes |

V3 made zero responder or judge calls; these are retrieval and packet
sufficiency measurements, not end-to-end answer accuracy. The preceding v2
[answer pilot](../10%20-%20Research%20Log/15%20-%202026-08-18%20-%20Policy-locked%201M-context%20answer%20pilot.md)
received 10/10 independent-judge decisions. V2 correctly derived four days,
three weeks, and 190 remaining pages, and returned the five concerts and six
museums in order.

These results show that the repaired development packet was sufficient for
those ten questions. They do **not** establish 100% general factual accuracy:

- the questions influenced treatment selection;
- the six-museum closure is selected-scope and explicitly non-global;
- only the preceding v2 treatment was provider-scored on ten development
  questions; and
- the required 100-question held-out campaign has made zero provider calls.

The later package/folder reorganization is implementation epoch v4 and is not
a retrieval-accuracy intervention. V3 evidence certifies commit
`bfa5b6daf6a5e61881ac10f0555e5d9972f9e1c2` and implementation SHA
`452be3bfa7524bb81676c7abcb032529a32a480311d24d1e17f8513c783ecd83`.
Because implementation identities hash source paths as well as bytes, v3
caches cannot certify the reorganized tree.

## Executable invariants

The implementation tests the proof boundaries independently of answer
accuracy:

- reachability and bounded association expansion:
  `test_hybrid_lexical_only_candidate_is_reachable`,
  `test_one_completed_interaction_can_recall_its_unique_outcome`, and
  `test_two_hop_read_balances_slots_across_frontiers`;
- selected-partition scan and snapshot validity:
  `test_partition_content_scan_is_exact_and_excludes_only_source_metadata`,
  `test_typed_partition_scan_forces_source_below_global_pool_at_fixed_count`,
  and `test_active_partition_snapshot_invalidates_when_transcript_advances`;
- scope and closure fail-open behavior:
  `test_routed_frontier_exhaustion_does_not_claim_hidden_partition_member`,
  `test_selected_partition_scope_opt_in_closes_without_global_claim`, and
  `test_post_coverage_closure_fails_open_on_any_unproven_gate`;
- event, role, scalar, and temporal identity:
  `test_performance_event_key_contracts_artist_detail_and_cross_source_recap`,
  `test_general_fixed_cardinality_uses_audited_role_reservation_tiers`,
  `test_query_compiler_treats_first_person_current_possession_as_scalar`, and
  `test_suppresses_only_proven_conflicting_approximate_recap`; and
- leakage, cache, and population provenance:
  `test_candidate_trace_joins_gold_only_in_offline_measurement`,
  `test_blind_causal_cache_prepare_never_embeds_held_out_questions`,
  `test_required_causal_cache_hit_is_read_only_and_reports_exact_pair`, and
  `test_locked_campaign_reconstructs_and_certifies_exact_population`.

These tests establish mechanical invariants. They do not replace the pending
held-out accuracy measurement.

## What was actually learned

The central lesson is that “retrieval quality” is not one scalar.

- Better embeddings help rank visible evidence.
- More routes help evidence become visible.
- Typed scans make bounded reachability auditable.
- Event identity prevents duplicate mentions from masquerading as coverage.
- Query compilation prevents the wrong retrieval algorithm from running.
- Role and temporal rules preserve who said what and which boundary is exact.
- Hard-budget packing determines whether selected evidence survives.
- Closure controls distractors only when its scope is honest.
- Receipts make the claim reproducible.

Any one layer can fail while the others look healthy. That is why answer
accuracy, source recall, value coverage, token cost, closure scope, and false
completion must remain separate metrics.

## Falsification and next use

The frozen treatment's next gate is the 100-question held-out validation
campaign under the same 8,000-token prompt-proxy cap. A 10-question canary
requires explicit authorization for exactly 20 central-dev calls; no call is
authorized by this theory note.

For diffuse questions, the same proof-carrying principles remain necessary but
fixed event grammars are not sufficient. The next design therefore adds
surprise-segmented episodes, temporal contiguity, discourse relations, query
obligations, and atomic evidence bundles. See
[`05 - EM-LLM Episodic Discourse Closure for Diffuse Retrieval.md`](05%20-%20EM-LLM%20Episodic%20Discourse%20Closure%20for%20Diffuse%20Retrieval.md).
