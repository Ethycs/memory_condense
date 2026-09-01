# R7 failure boundary and closure-aware semantic completion

**Date:** 2026-08-29

**Status:** architecture decision from frozen development results. V5's full
selector and judge lifecycle is sealed and replayed at **88/100**, so V3
remains the protected 89/100 parent. V6 source-local reinjection and V7 global
typed search are active. This is not a 95% claim and must not be used as
confirmation evidence.

## Decision in one sentence

Keep the cumulative V3 memory answer as protected state, use an LLM only to
select between exact frozen answer candidates under closure-aware grounding,
reinject the source-local user/episode neighborhood around selected evidence,
and reserve global semantic search for rows that still lack a complete proof.

## Evidence that fixes the boundary

The replay-verified V3 result is 89/100. R7 repaired the semantic-residual
apparatus itself: 68/68 eligible questions receive bounded prompts, no
eligible row falls back for packing, the largest complete answer envelope is
5,544/8,000 tokens, construction/replay make zero provider calls, and no
transformer-token state is retained. The full R7 lifecycle nevertheless
scores 88/100.

That negative result separates four loss stages:

| Stage | Frozen observation | Consequence |
| --- | --- | --- |
| candidate validation | q49 contains the exact Denver/music preference and Terra emits a relevant answer, but a lexical-subset rule rejects it | replace lexical entailment with semantic selection over exact candidate strings |
| source-local projection | q40 and q67 reach labelled source records but expose generic assistant tails rather than the neighboring user acquisition/visit assertions | reopen the selected source group or episode before asking another global retriever |
| global discovery/ranking | q14, q28, q53, and q54 omit answer-bearing source groups before packing; q69 omits one required clothing event | route only these unresolved obligations into global semantic search |
| data/eval boundary | q94 retains the known source-date/reference inconsistency | report it; do not weaken a generic temporal policy to fit it |

R7 changed only three V3 predictions. Two became Sol regressions (31 and 51),
one remained correct (50), and no changed row rescued a V3 miss. The apparent
one-point offset came from a fresh Sol flip on byte-identical ordinal 82. The
new architecture must therefore protect V3 more strongly while still allowing
real residual repairs.

## The cumulative pipeline

```text
immutable common memory store
  -> existing typed V3 retrieval/operators
  -> R7 bounded residual selection (R plane)
  -> protected owner evidence (P plane)
  -> V5 closure-aware exact-candidate selector
       -> accept exact candidate
       -> keep exact V3 answer
       -> needs_global_search
  -> V6 source-group / episode-neighbor reinjection (L plane)
  -> V7 semantic global-to-local search only if obligations remain
  -> typed deterministic execution + bounded synthesis
  -> exact replay and independent judge
```

Each layer consumes the previous layer's sealed output. No layer silently
replaces the store, the retrieval stack, or the protected answer population.

## V5: semantic selection, not unconstrained regeneration

V5 authenticates the exact R7 preflight and byte-identical answer/replay. It
mechanically freezes every raw Terra `replace` completion, including candidates
that V4 accepted and candidates its lexical rule rejected. The verifier sees:

- the dated question;
- the exact protected V3/current answer;
- the exact frozen candidate;
- the complete bounded R and P evidence planes, with role, event time, source
  group, quote hashes, and source receipts;
- the candidate's original citations;
- the typed operator program; and
- the R7 frontier state, including unresolved-survivor and closure receipts.

The verifier selects an existing string; it never generates answer text.
Malformed, unsupported, or ambiguous output fails closed to V3. Exact-current
Terra responses that merely supplied handles are canonicalized to V3 with
empty citations.

Semantic grounding is not permission to promote a local subset into a global
answer. If `packing_closed=false` or `support_closure_proven=false`, a new
scalar or set for a question asking `all`, `total`, `how many`, or equivalent
is forbidden unless a separately sealed typed operand-closure proof covers the
requested predicate. A locally correct subset count produces
`needs_global_search`, not an authoritative total.

R and P keep separate non-borrowable budgets. R remains at or below 2,400
serialized tokens; P retains its own exact owner cap. The full prompt plus
output reserve remains at or below 8,000 tokens. This prevents provenance
metadata from being discarded merely to force two logically different planes
under one artificial subtotal.

The sealed V5 preflight mechanically contains all 15 raw R7 `replace`
candidates and separately receipts all 13 exact-current completions that had
carried handles. Its 15 Sol prompts are unique. Maximum complete envelope is
7,742/8,000 tokens with a 768-token output reserve; maximum exact R and P
planes are 2,393 and 793 tokens respectively. The full enriched R/P union is
retained and separately reported (maximum 4,581 tokens), rather than being
misrepresented as a single 2,400-token plane. All 15 R7 frontiers are open.
Eleven candidates have a question-only typed specification requiring complete
frontier and therefore deterministically fall back to current plus a search
trigger in V5; local arithmetic execution alone is not treated as proof that
all requested operands were retrieved. The other four remain eligible for
bounded direct or preference-synthesis selection, subject to the strict
role/personal-scope verifier.

### Measured V5 result

V5 made exactly 15 Sol selector calls with zero retries. It selected the exact
frozen candidates for ordinals 36, 49, and 81, canonicalized ordinal 6 as an
equivalent current answer, and emitted deterministic search triggers for the
eleven questions whose open frontier requires complete evidence. The answer
run and replay are byte-identical at
`3645a869bbee3835f1e9bc3a8c1d7104738bbbcd5266c319fa23138e77ed02c7`.

The independent full-100 judge also completed with zero retries and replayed
byte-identically. V5 scores **88/100**. Relative to V3, ordinal 36 regresses,
ordinal 49 changes but remains incorrect, and ordinal 81 remains correct. It
therefore yields zero rescues and one regression. The selector is retained as
a closure-aware safety/router diagnostic, but its answers are not promoted;
V3 remains protected. Full receipts and hashes are in Research Log 83.

## V6: local-to-global and global-to-local linking

Several misses do not need another global search. A selected source ID or
episode already points to the right memory neighborhood, but segment ranking
chooses an assistant explanation, generic advice tail, or partial sentence
instead of the user assertion that made the source relevant.

V6 therefore adds a separate, bounded local linking plane after R selection:

1. freeze the R selection;
2. resolve each selected source-group handle to its exact immutable source;
3. retrieve the user-role parent/sibling spans and a small source-local
   temporal shell or compiled episode;
4. rank within that neighborhood for requested entity, action, date, and
   numeric operands;
5. deduplicate against R and P only after this independent selection; and
6. pack the novel local rows under their own non-borrowable budget before the
   terminal 8,000-token fitter.

This is the practical form of provenance/CAV reinjection: the global hit names
the local group, then exact local evidence is written back into the answer
frontier. It adds text only through immutable provenance; a CAV guide itself
does not become answer evidence. Existing `hybrid_neighbor` and episode-store
primitives should be reused rather than constructing a new graph authority.

The expected causal targets are q40 and q67, where source IDs are right but
the answer-bearing user spans are wrong, plus the missing blazer operand in
q69 and possibly the personal UberEats statement in q97. This expectation is
development-contaminated and can only guide the mechanism assay, not the
confirmation claim.

### Measured V6 boundary

The provider-free V6 primitive and exact R7 adapter are now implemented. The
sealed reduced-ten construction and full upstream replay are byte-identical at
`84664375e9453db85870697669eeef0bf7b1a23ee649af1a26b57dee800ec954`.
Across the ten explicit diagnostic rows, V6 produces 62 novel L spans, closes
111 post-selection duplicates to exact R/P owners, and leaves 215 selected
rows outside the independent 1,200-token L caps. All ten local typed-obligation
sets are covered, but every packing frontier remains open and therefore still
requires global search. These are retrieval-plane measurements, not an answer
score; the construction is gold-blind and makes zero provider calls.

The q82 episode control identifies a narrower boundary. Both selected handles
map to the correct immutable source. Their two direct fixed-interval episodes
contain the exact user Garmin assertion and the exact dated chain/cassette
performance assertion, respectively. Nevertheless, a 64-row globally ranked
selection truncates and a 1,200-token pack admits generic neighboring text
instead of either target. Sentence-level handling of mixed fact-plus-question
turns repairs one scoring defect but does not change that exact-store outcome.

Thus V6 demonstrates global-to-local connectivity, but connectivity alone is
not evidence compilation. The next terminal compiler must reserve the best
user-owned assertion per directly anchored episode and per unresolved typed
obligation before allowing generic assistant context or other groups to spend
the remaining L budget. R, P, and L stay separately accounted and
non-borrowable; exact deduplication stays after selection; an open local or
global frontier remains an explicit V7 trigger. Research Log 84 contains the
artifacts, per-ordinal pressure table, topology receipts, and tests.

### Measured V6.1 closure of the local compiler defect

The proposed direct-episode reservation has now been implemented and measured.
V6.1 treats source-neighbor and direct/adjacent episode populations as separate
lanes, preserves authenticated seed order, promotes exact factual/action/date/
numeric assertions inside direct episodes, and gives every direct factual
candidate an independent head before contextual tails. Selection still occurs
before protected-evidence deduplication: a protected anchor is audited and
closed to its exact R/P owner, but it no longer suppresses the next novel user
assertion in that episode. The lanes interleave deterministically and
skip-and-continue beneath the unchanged non-borrowable 1,200-token L cap.

The frozen identifiers are
`source_group_episode_neighbor_reinjection_v1_1` and
`stratified-direct-assertion-lanes-post-dedup-skip-continue-v3`; the default
policy receipt is
`c15f430054445dc96c246a3ba156710a6990b807c07dea571f042104a52795df`.
On the exact q82 store, the construction and outer replay are byte-identical at
`ca9c97b5678be6ece8e33365ceea0be5698fbfcee8c8993cbc4872f685311676`.
The L plane now includes both exact user assertions: the new Garmin bike
computer from direct episode `episode-f43e2711dea9ce00e405e21f`, and the dated
chain/cassette replacement and performance improvement from direct episode
`episode-517b033bcde19f8a88a83f51`.

This is a local-recall success, not a closure proof. The row packs 12 novel
spans at 1,176/1,200 tokens, closes 11 protected duplicates, leaves 41 selected
spans budget-unpacked, and truncates selection at 64. Consequently
`needs_global_search` remains true and V7 still receives the unresolved global
frontier. Focused V6.1 tests pass 10/10 and the adjacent mechanism/adapter suite
passes 53/53, with zero provider calls and zero retained transformer token
state. Research Log 84 records the full before/after artifacts and receipts.

## V7: the actual semantic binary-search fallback

R7's deterministic classifier is deliberately fail-open. On nine of the ten
stable misses it prunes zero leaves, retains roughly 700 cells, and ranks about
7,300--7,600 exact segments into a 9--19-row residual packet. It is a complete
conservative scan followed by top packing, not an effective semantic binary
search.

The final lane must reduce the unresolved population before packing:

1. compile the typed evidence predicate and high-recall query facets;
2. exclude only structurally impossible cells (wrong role, impossible event
   interval, exact-literal contradiction) deterministically;
3. show a gold-free semantic classifier bounded node/cell descriptors and let
   it return only `definitely_no` or `may_answer`;
4. descend every `may_answer` branch and preserve an exact leaf partition;
5. expose answer-bearing raw spans from retained cells, then perform the V6
   source-local reinjection;
6. for global count/set questions, continue until every retained operand is
   classified and deduplicated rather than stopping at an ordinary top-k; and
7. keep the frontier explicitly open if classifier or budget limits prevent a
   complete scan.

Query-time phrase expansion and the existing BM25/BGE indexes may prioritize
nodes, but they cannot authorize absence. The semantic classifier's decisions,
node receipts, leaf outcomes, and exact source projections must be checkpointed
and replayable. Intermediate prompts and the terminal answer prompt each obey
the hard cap; no request-derived hidden state, residual stream, attention map,
or K/V cache is persisted.

## Generic routing rule

No ordinal list is permitted in production policy. A row reaches V7 when any
of these receipt-bound conditions holds:

- V5 returns `needs_global_search`;
- a global count/set operator lacks operand closure;
- a required typed slot has no answer-bearing user-role witness;
- a direct dated-event query lacks a matching event/entity witness; or
- V6 resolves a selected group but still cannot satisfy the operator proof.

A supported exact current answer, a supported exact candidate, or a complete
typed proof stops the pipeline. This makes runtime cost adaptive without
letting confidence or score labels change retrieval authority.

## Evaluation gates

The development sequence is:

1. seal V5's provider-free preflight and tests;
2. run its exact candidate-verifier call population, materialize, and replay;
3. judge the complete 100 rows, not only changed rows;
4. run a reduced structural V6/V7 assay on generic trigger outputs and measure
   target reach only after construction freezes;
5. promote the same trigger and budgets to full 100 and rerun the full judge;
6. freeze policy only at or above 95/100; and
7. evaluate the untouched confirmation population, reporting both the full
   set and the predeclared non-exposed sensitivity slice.

The fair Mem0 arm remains independent: same ten approximately one-million-token
namespaces, question order, 8,000-token answer envelope, reader, judge, neutral
fallback, and separate write/search/answer/judge accounting.
