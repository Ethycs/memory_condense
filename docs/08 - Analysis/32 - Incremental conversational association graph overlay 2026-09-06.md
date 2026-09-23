# Incremental conversational association graph overlay

Date: 2026-09-06

Status: provider-free graph core, persistent append-at-ingest T1g lifecycle,
bounded historical-store bootstrap, and the authenticated resident story read
path are implemented. The sealed v4 full100 structural assay reaches 99/100
complete reference-source coverage; this is not a HippoRAG, Graphiti, or judged
answer-accuracy result. Production `build_context` composition and optional
semantic enrichment remain open.

## Decision

Build a small, additive **conversation-time association overlay** behind the
existing fast retrieval lanes. Borrow HippoRAG 2's useful separation between
raw passage memory and a sparse association index, and borrow Graphiti's
incremental episode/provenance/validity model. Do not replace the common raw
memory store, the typed operators, or the current provider-free fast path, and
do not adopt either package as the production store before a matched assay.

The first version should be deterministic and provider-free:

1. append the raw user or assistant turn as the factual authority;
2. compile bounded phrase, entity-like surface, action, date, role, source, and
   turn-adjacency addresses as that turn arrives;
3. connect new addresses to existing exact or normalized addresses;
4. seed a bounded graph walk from the ordinary lexical/typed/source hits;
5. hydrate only exact raw chunks reached by the walk; and
6. let this lane select under its own budget before global exact-ID dedup.

An optional local LLM/OpenIE enrichment stage can follow asynchronously. It
must never delay durable capture, become the sole route to a fact, or emit an
answer-bearing graph claim without exact source spans.

## What the published systems actually contribute

[HippoRAG](https://arxiv.org/abs/2405.14831) introduced the useful retrieval
pattern: an LLM constructs an open knowledge graph, a retriever links query
cues to graph nodes, and Personalized PageRank performs single-pass
associative retrieval. [HippoRAG 2](https://arxiv.org/abs/2502.14802) corrects
the first version's concept/context loss by placing both phrase and passage
nodes in the graph. Its indexing path extracts OpenIE triples with an LLM,
adds embedding-threshold synonym edges, and connects each passage to its
phrases. Online retrieval embeds the query against triples and passages, uses
an LLM recognition filter over candidate triples, then ranks passage nodes by
PPR.

That is evidence for the **shape** of an associative overlay, not for replacing
our stack. HippoRAG 2's own error analysis says triple filtering and graph
search are its two main error sources. Eighteen percent of its analyzed misses
had no triples left after filtering, and graph search could still omit support
after the correct phrases were linked. Its MuSiQue accounting reports 9.2M
input and 3.0M output indexing tokens for 11,656 passages, 99.5 minutes of
indexing, 1.2 seconds per query, and 9.9 GB of QA-side memory. The current
[official implementation](https://github.com/OSU-NLP-Group/HippoRAG) does
support incremental calls to `index(docs)`: it extracts only missing chunks
and updates graph state, with regression tests for sequential indexing and
deletion. It is nevertheless a document-indexing library; it does not capture
our conversation stream for us.

[Graphiti](https://github.com/getzep/graphiti) is closer to the desired write
lifecycle. It treats an incoming message or record as an episode, maintains
source provenance and bi-temporal fact validity, and incrementally updates a
hybrid keyword/semantic/graph index without a complete batch rebuild. Its
official setup, however, requires a graph backend and defaults to LLM and
embedding calls; its documentation warns that smaller/non-structured-output
models can produce schema failures. It is therefore a useful reference for
incremental and temporal semantics, not a zero-cost drop-in dependency.

## What already exists here

The project is much closer to this design than a package comparison suggests:

- `turns` and `chunks` are already the append-only, source-identified raw
  memory authority.
- BM25, dense addresses, typed witness postings, temporal enumeration, and
  source-neighborhood coordinates already supply query seeds.
- `episodes`, `episode_evidence`, `discourse_units`,
  `discourse_relations`, and their exact span tables already represent
  provenance-bound episodic and relational artifacts.
- `chunk_head_edges`, `hebbian_chunk_edges`, and `consolidation_edges` already
  store sparse learned associations without storing token state.
- `diffuse_association_heat` already implements a bounded, restartable graph
  diffusion primitive; a new dependency on `igraph` is not required to test
  the hypothesis.
- `RuleBasedDiscourseLinker` already emits conservative units, source-local
  sequence edges, and explicit-cue relations.

Raw chunks should remain the neocortical/content store in HippoRAG's analogy.
The new graph is only an address plane. A phrase node, synonym link, CAV, or
summary never replaces its cited raw evidence.

## The actual gaps

### 0. Two existing append paths are not actually incremental

`DiscourseStore.publish()` currently recomputes discourse content digests by
streaming the source and graph row sets. Publishing one new graph delta is
therefore `O(N + G)` in the current store even when extraction and linking are
bounded. `SourceNeighborhoodIndex` is also immutable and re-sorts/rebuilds the
complete metadata population. A production conversation-time overlay must
replace those two lifecycle costs with an append-only revision/Merkle delta
and an incrementally updated source tail/index; moving either full rebuild to
a background worker would not make ingestion incremental.

### 1. No useful cross-publication phrase hub

`RuleBasedDiscourseLinker` creates one unit for an entire input atom and derives
its canonical key from the first twelve non-stop terms. It does not extract
multiple phrase occurrences from a turn. Its nearest-prior relations are also
computed only among units passed to the same `link()` call.

More importantly, `link_and_publish_discourse()` currently requires every
relation member to be emitted in that same linker output. A linker invoked for
one new turn therefore cannot link its new unit to a unit published for an old
turn, even though `DiscourseStore.publish()` itself can persist a caller-built
relation that references existing units.

This is the main online-memory seam to add.

### 2. Immutable evidence needs a virtual canonical layer

`discourse_units` are immutable and evidence-bound. Reusing one stable phrase
unit and appending new evidence to it would violate that identity contract.
The safe first form is:

- one immutable phrase-occurrence unit per exact evidence span;
- `canonical_key` as the shared virtual hub indexed by the existing
  `(artifact_id, kind, canonical_key, asserted_ordinal)` index;
- explicit sparse relations only where direction or semantics matter; and
- a rebuildable resident posting from canonical key to occurrence/unit/chunk.

This avoids mutable graph claims and avoids quadratic all-pairs synonym edges.
An alias table or embedding-derived alias cache can be added later as a
versioned derived artifact.

### 3. No per-turn enrichment watermark

The discourse coverage receipts can prove that a static artifact processed a
source snapshot, but the live path needs an explicit high-water mark and
status for each new chunk: `pending`, `deterministic_ready`,
`semantic_ready`, `failed_retryable`, or `no_output`. The raw turn must be
queryable immediately, even when semantic enrichment is behind.

### 4. Graph traversal was not composed as an independent lane

The original hot-v3 store did not seed a phrase/passage walk. The v4 assay now
composes an authenticated resident graph/story resolver behind the typed lane,
with exact upstream candidate identity, a per-witness and aggregate token cap,
and post-selection dedup. It returns the exact baseline object on semantic
abstention. The corresponding production `build_context` lane remains to be
wired; when it is, graph candidates must retain an independent budget and must
not pre-filter or erase BM25, typed, temporal, profile, source-neighbor, or
parent raw candidates.

## Proposed write pipeline

```text
conversation turn
  -> T0 atomic raw turn + pending-manifest/chunk-reservation capture
       -> durable and retryable, but not yet BM25/dense searchable
  -> T1 batched chunk materialization + BM25/dense index commit
       -> ordinary raw retrieval becomes searchable here
  -> T1g bounded deterministic graph compile
       -> phrase occurrences + action/date/role/source addresses
       -> previous/next turn and exact-canonical-key links
       -> graph revision + per-turn terminal receipt
  -> T2 optional asynchronous semantic enrichment
       -> OpenIE/entity aliases/revision or validity relations
       -> exact evidence spans + producer/checkpoint identity
  -> periodic maintenance
       -> degree pruning, alias-cache rebuild, community/hub candidates
```

T0 is the durability boundary and T1 is the ordinary-searchability boundary.
T0 must not wait for either indexing or graph compilation. T1g should normally
complete faster than answer generation, but graph lag must fail open to the
ordinary T1 indexes. T2 may use the local LiteLLM endpoint or an on-device
extractor only after its exact model, call count, payload, and artifact contract
are separately authorized and sealed.

For each appended turn, T1g work must be bounded by the new turn and a capped
posting neighborhood, not by the full corpus. Exact-key insertion is expected
`O(p log N)` for `p` extracted occurrences. Alias discovery should query a
small ANN/posting frontier and enforce a degree cap; it must not compare every
new phrase against every historical phrase.

### Production persistence seam

The capture-first implementation already exposes the safe seam. Inside
`IngestWorkflowMixin._publish_staged_turns()`, T0 should claim a graph compile
obligation in the same transaction that publishes the turn and pending-ingest
manifest. That claim is only a small journal insert; it must not tokenize,
extract phrases, or publish graph state. After `drain_pending_ingests()` has
committed chunks, BM25 terms, and the ANN mapping, the scheduler may run a
separately bounded T1g worker. A T1g exception must never roll back T1 or make
the raw turn unavailable.

A dedicated versioned artifact is safer than overloading discourse snapshots:

- `graph_artifacts` binds implementation and extraction/traversal policy;
- `pending_graph_compilations` gives every captured turn a terminal
  `pending`, `ready`, or `no_output` receipt plus retry state;
- `conversation_graph_chunks` stores physical coordinates, content hashes,
  append revision, and receipts but no duplicate factual text;
- `conversation_phrase_occurrences` stores canonical keys and exact source
  spans, indexed by `(artifact_id, canonical_key, append_revision, chunk_id)`;
  and
- `conversation_graph_state` publishes the current revision and ready counts.

Compile from immutable turn/chunk manifests outside the SQLite writer lock,
then revalidate and commit occurrences, the append receipt, graph revision,
and the pending-to-ready CAS in one short transaction. A restart either pages
the durable graph once or loads only rows newer than its observed revision; it
must never re-extract the complete corpus. Backfilled old event ordinals still
receive a new append revision and locally splice the source predecessor and
successor indexes. Per-turn terminal receipts, not a scalar maximum ordinal,
are the completeness authority.

## Proposed read pipeline

```text
dated question
  -> current hot lexical + typed + profile retrieval
  -> seed chunks and structured cues
  -> canonical phrase/alias lookup
  -> bounded heat/PPR-like diffusion over activated subgraph
  -> ranked raw chunk IDs with path receipts
  -> hydrate exact source text
  -> graph-lane selection under its own token budget
  -> global post-selection exact-ID dedup and source protection
  -> primary LLM prompt
```

Use graph expansion only when it has a material seed. A no-seed or low-support
graph lane fails open to the unchanged parent packet. Traversal should begin
with a small activated subgraph rather than running global PPR on every prompt:
query seed postings, their bounded phrase/alias neighborhoods, and linked
passages are enough to test local-to-global connectivity while keeping online
latency predictable.

The path receipt for every admitted graph candidate should include:

- query/seed identity;
- artifact and source-snapshot identity;
- ordered unit/relation or virtual-hub path;
- edge types, weights, decay coordinates, and traversal budget;
- destination chunk and exact evidence-span identity; and
- selection-before-dedup and final ownership outcome.

## Relationship to CAV, Hebbian heat, and episodes

This layer composes the existing mechanisms instead of replacing them:

- phrase/canonical-key links solve **cold-start semantic connectivity**;
- source adjacency and episodes solve **local sequence continuity**;
- Hebbian co-access learns **usage-dependent associations** after retrieval;
- CAV/QK/OV links contribute **model-observed concept transport** when their
  artifact is present;
- event-time and typed postings solve **date/action/operator routing**; and
- the graph walk transports activation among those address types before raw
  hydration.

In particular, do not let Hebbian observations train on candidates admitted
only by the same graph traversal in the same tick. Preserve the existing rule
that graph-admitted results cannot immediately reinforce themselves.

## Measured implementation status

The first local building blocks now exist. The provider-free
`IncrementalConversationGraph` appends exact phrase occurrences, turn/chunk
coordinates, source-local sequence edges, and bounded same-source recurrence.
It keeps sequence slots outside phrase-degree caps, supports deterministic
best-path relaxation over at most two hops, and emits seed/path provenance. The
hot graph adapter requires explicit seed bindings, so an ungrounded bare chunk
ID cannot silently enter the walk.

The implementation is intentionally smaller than HippoRAG or Graphiti. It has
no LLM OpenIE linker, embedding synonym layer, external graph database, or
global PPR. Exact phrase, sequence, and bounded source-story addresses now
compile incrementally after ordinary search becomes durable. T0 journals only
a manifest-bound obligation; a separate fail-open T1g worker publishes
immutable deltas, terminal per-turn receipts, and a SHA-256 checkpoint chain.
Raw text remains in the authoritative turn/chunk tables rather than being
duplicated in the graph. Restart hydrates the persisted delta log without
re-extracting phrases, and an already resident process adopts only revisions it
has not seen.

The restart and backfill boundary is now explicit. A bounded
`bootstrap_conversation_graph(max_turns=32)` call claims only indexed turns
whose original ingest manifest can still be authenticated, is hard-capped at
512 turns per call, and reports both remaining and unsupported legacy turns.
Manifestless legacy rows are never converted into invented provenance. Graph
state and deltas have no-delete guards, normalization is idempotent across
append/restore, and resident catch-up pins one target revision so a concurrent
writer cannot move the hydration target mid-read.

The optimized ordered-story index changes the q86 cost materially. Over 7,498
chunks and 488 sources, the first cold build took 27.523 seconds; that is now
ingest/rebuild work rather than required query work. Existing graph plus story
indexing averaged 4.987 ms per chunk. The core q86 story query took 3.332 ms,
and 200 authenticated wrapper replays measured 5.654 ms mean and 7.086 ms p95.
All returned sources still hydrate exact raw evidence and retain sealed seed,
path, policy, and graph-revision receipts. The v4 evaluation read path now uses
this authenticated resident graph resolver. Production `build_context` still
needs the equivalent independently budgeted lane.

A separate strict ordered-story repair converted the useful source connectivity
into an admitted evidence result. On ordinal 86 it selects, without gold, Muir
Woods -> Big Sur/Monterey -> Yosemite in the requested chronological order.
In the final v4 full100, source recall moves from 2/3 to 3/3, best F1 from
0.400 to 0.483, and the packet remains at 66 packed rows. The result retains
the three physical source chunks rather than graph summaries. The original
reduced assay and its admission gates are recorded in [Research Log 115](../10%20-%20Research%20Log/115%20-%202026-09-06%20-%20Gold-blind%20ordered-story%20residual7%20repair.md).

The final v4 full100 transfer completed on 2026-09-07. Relative to the original
hot-v3 parent, complete reference-source reach is **99/100 versus 93/100** and
literal-answer hits are **57/100 versus 56/100**. All 100 arms replay exactly.
Construction and replay load no gold and make no model or provider calls; gold
is loaded only by the sealed post-hoc scorer. Ordinal 93 is now source-complete
after a bounded `buisiness` -> `business`, near-date, completed-first-person
business-milestone route: source recall moves from 0/2 to 2/2 and best F1 from
0.207 to 0.467. This is evidence construction coverage, not answer accuracy,
and it does not supersede the latest 70/100 judged fast-path answer result.

The run keeps ten memory namespaces resident and builds the story graph for
only two question-eligible namespaces, not all ten. Those graph builds total
56.294 seconds. Warm question processing is 139.164 ms mean and 337.139 ms
p95; prompt-ready composition is 77.783 ms mean and 101.348 ms p95. Ordinal
86 is 135.938 ms end to end and ordinal 93 is 74.107 ms. The full cold assay is
379.163 seconds because it reconstructs all ten sealed one-million-token test
stores; this is not the live-turn append or resident-query latency.

## Next falsifiable assay

Do not start by re-indexing full100 with LLM OpenIE. The only strict source miss
is ordinal 77: it already contains the literal answer and the two semantically
necessary museum sources, while the absent third reference source is an
auxiliary lecture comparator. Adding that source would optimize a label-level
diagnostic rather than demonstrated answer sufficiency and could displace more
useful evidence.

The live uncertainty is now synthesis. Freeze the 30 questions that failed the
latest Sol-judged answer run, generate answers only for those new sealed v4
packets with one universal question-only operation-aware responder instruction
(typed shape plus temporal/comparison/set operation, no gold), and judge those
sealed predictions. This 30 Terra + 30 Sol screen directly asks whether the
new evidence closes the earlier 70/100 answer result. Require at least 25/30
recoveries before paying for a full100 confirmation, and do not route or tune
by ordinal or failure label. A full100 confirmation is still required to rule
out regressions among the prior 70 successes.

Only failures whose necessary raw evidence is still absent after that assay
justify a HippoRAG-style semantic-linking shadow. It must add unique evidence
under the same source/span/token receipts before LLM OpenIE, embeddings, PPR,
or an external graph backend enters the default path.

## Implementation order

1. **Implemented:** deterministic phrase-occurrence extraction,
   resident phrase/chunk postings, bounded two-hop expansion, typed seed
   bindings, and exact path receipts.
2. **Implemented and admitted:** strict ordered-story replacement for
   cardinality-known, ambiguous temporal lists, including residual7 and the
   provider-free full100 structural transfer.
3. **Implemented:** T0 graph-obligation capture and immutable phrase,
   sequence, and source-story delta persistence in a separately fail-open T1g
   worker after T1 becomes searchable.
4. **Implemented in the v4 assay:** authenticated resident graph/story query
   composition, strict protection, post-selection dedup, and question-gated
   graph construction. Production `build_context` remains separate.
5. **Implemented:** explicit bounded bootstrap for manifest-backed indexed
   pre-v16 turns. Add a serialized snapshot only if first-process O(G)
   hydration is material.
6. **Next measurement:** operation-aware answer synthesis and judgment over the
   30 previously failed questions, followed by full100 only on admission.
7. Add cross-publication semantic aliases plus event validity and supersede
   edges for temporal/update questions.
8. Compare optional asynchronous local-LLM OpenIE against the deterministic
   phrase graph on the remaining graph-eligible failures.
9. Consider an external Graphiti/HippoRAG backend only if the local graph's
   storage or maintenance—not retrieval quality—is the measured bottleneck.

Production work must still compose the existing indexed canonical-key lookup
and resident adjacency into `build_context`, expose graph scheduler lag/backoff
telemetry, and add temporal validity coordinates. QK/CAV, Hebbian, and
consolidation edges may seed or weight the walk, but remain soft, separately
receipted transitions rather than semantic fact authority.

## Bottom line

Yes, memory should be linked as the conversation happens. That write lifecycle
now exists locally: exact raw turns remain authoritative while a persistent,
provider-free T1g address graph accumulates bounded phrase, sequence, and story
links. The q86 graph lookup is millisecond-scale once resident, and the sealed
v4 full100 reaches 99/100 strict source coverage without retrieval-time model
calls. The next step is the 30-question synthesis/judgment screen and production
`build_context` composition, not a full HippoRAG migration. Optional semantic
linking belongs in asynchronous T2 only after it demonstrates unique gains over
this deterministic baseline.
