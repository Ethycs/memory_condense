# Orphaning audit and lifecycle repair

**Date:** 2026-09-01

**Status:** implemented in the working tree; combined core and semantic
verification passing; provider-free; no new answer score or 95% claim

## Decision

Treat reachability as a cross-layer contract, not as a property that each
storage or retrieval mechanism can establish independently.

The operative invariant is:

> Every producer output has one durable owner, and every consumer-visible
> identity resolves either to that owned object, to an explicit successor or
> retirement record, or to an authenticated omission receipt. Publication,
> retirement, and index visibility may not leave an unlabelled gap between
> those states.

In shorthand:

```text
producer -> owner -> consumer
             |
             +-> explicit successor / tombstone / omission receipt
```

The audit applied this invariant to turns, chunks, dense and lexical indexes,
association artifacts, extracted memories, correction chains, provenance, and
sealed retrieval constructions.

## Three kinds of orphaning

### Data-loss orphaning

The canonical data is absent or no longer reachable even though the operation
appeared to succeed. Examples are retiring an old memory before its replacement
exists, or publishing a turn whose chunks failed to enter the indexes. This is
the highest-severity class because no downstream ranking or larger prompt can
recover what the lifecycle discarded.

### Traceability orphaning

The payload may still exist, but an identity, edge, provenance citation, or
sealed mechanism receipt no longer explains how to reach or trust it. A ghost
`chunk_id`, a duplicate merge that loses one predecessor edge, and repaired
semantics published under a historical format identity are all traceability
failures. They undermine replay and diagnosis even when immediate answer
recall happens to survive.

### Cold retention

Obsolete or zero-degree derived rows remain in storage after they cease to be
useful. The observed cold Hebbian/consolidation nodes and older association
artifacts are primarily maintenance and cost debt: the live retrieval paths do
not currently treat them as answer evidence. They should be compacted, but
they are not equivalent to a demonstrated recall loss and were not given the
same priority as the first two classes.

## Confirmed defects

### Memory supersession could strand the predecessor

The old path committed `active -> superseded` before embedding and inserting
the replacement. A synthetic embedding failure reproduced one superseded row,
zero active rows, and zero successors. The public `create(...,
supersedes=...)` path could also manufacture a correction pointer without
proving that its target was an active item.

Exact-content coalescence exposed a second shape problem. A replacement has
one scalar `supersedes` field, but it can absorb provenance from more than one
active predecessor. Overwriting a duplicate's existing scalar chain would
make one history look complete by orphaning another.

Pre-v12 `dedupe_existing()` rows also used that scalar in the opposite
direction: the retired duplicate pointed at the survivor. Reading every scalar
literally as a backwards revision edge therefore returned a retired duplicate
as a successor of an active survivor and missed the forward path from loser to
survivor. Equal timestamps made chronology alone insufficient to recognize
the legacy layout.

Finally, a read-only `MemoryStore` could enter an embedding path before SQLite
rejected the mutation, repeating the ingest-side problem of performing a
fallible external side effect for an operation that could never commit.

### Ingest published ownership before fallible work completed

Single-turn ingest inserted the transcript row before embedding. A reproduced
embedding failure therefore left one turn and no chunks; retrying the same
explicit ID then failed at the transcript boundary. Batch ingest had the same
publication-order hazard, plus a partial-publication window when a later append
failed.

Index construction is multi-representation. A failure after the dense SQLite
write could leave the transcript/chunk owner population inconsistent with BM25
or the live HNSW process unless compensation knew exactly which rows belonged
to the failed attempt.

One batch-specific identity collision was also reproduced. Repeating the same
explicit turn twice is transcript-idempotent and gives its chunks the same
deterministic IDs, but flattening both copies for indexing could allocate two
HNSW labels for one chunk identity. When both records omitted `created_at`,
each staged `Turn` also generated a different timestamp, so an otherwise exact
duplicate could be rejected before it reached that physical-chunk deduplication.

Moving embedding before publication also exposed an ordering requirement for
read-only stores. The first staged implementation called the embedder before
SQLite rejected the eventual write, so a rejected compiled-cache ingest still
performed fallible and potentially expensive external work.

### Chunk index mutation crossed commits and process boundaries

Dense columns, BM25 postings, CAV/head artifacts, Hebbian rows, and
consolidation triggers were retired through separately committing helpers. A
failure between them could leave a nominally retired chunk retrievable by a
surviving representation.

Addition had the dual failure. Dense chunk state committed before BM25, so a
lexical failure could strand a dense-only row. Compensation removed that row
only when the turn itself was new; retrying an already-present explicit turn
could therefore retain the half-indexed identity and skip the repair on its
next attempt.

Cross-process synchronization compared only durable and locally mapped chunk
counts. That could detect simple growth, but not deletion or an add/delete swap
with the same population size. A live retriever could consequently retain a
deleted label or miss another process's replacement.

### Provenance accepted unresolved chunk identities

The validator authenticated a quote against its turn but did not require an
optional `chunk_id` to exist or to contain that quote. It therefore admitted
citations whose local chunk owner was missing, belonged to another turn,
disagreed with its stored character span, or did not contain the cited text.

### A historical migration receipt borrowed the ambient version

The v11 discourse-snapshot post-migration hook used
`CURRENT_SCHEMA_VERSION` when sealing its baseline. Once v12 existed, replaying
the v11 hook could claim schema 12 before the v12 migration had run. This was a
traceability defect in the migration boundary, not a content-recall result.

### Repaired retrieval behavior was not consistently reachable or versioned

The semantic-global terminal adapter already contained linked and post-dedup
backfill successors, but the construction path did not expose all of them as
an explicit production choice. Separately, typed-additive composition had
evolved from its original dedup-before-lane behavior while retaining the v1
top-level identity. In both cases, an implementation could exist without a
consumer selecting it, or replay could not tell which semantics produced an
artifact.

### Lower-priority trace gaps remain

The audit also found candidate-level explainability gaps around CAV/QK
hydration, hard caps, and some dropped typed bindings. Aggregate closure and
omission records exist, but they do not always preserve a direct
mechanism-to-candidate ownership receipt. These are traceability improvements,
not reproduced data deletion, and are intentionally left for a subsequent
receipt pass.

## Implemented repairs

### Atomic memory correction and many-to-one successor history

Supersession now resolves the fallible embedding before mutation, then uses one
`BEGIN IMMEDIATE` transaction to:

1. prove and retire the active predecessor;
2. insert a fresh replacement with the direct scalar back-link;
3. merge provenance from any unrelated active exact duplicate;
4. retire that duplicate without changing its existing scalar history; and
5. write an explicit duplicate-to-successor redirect.

Schema v12 adds `memory_successor_redirects` for that last many-to-one edge.
`MemoryStore.successors()` returns the union of ordinary scalar successors and
these additive redirects. `dedupe_existing()` uses the same redirect model,
and public creation with `supersedes` now routes through the atomic lifecycle
and rejects missing or non-active targets.

For stores that already contain pre-v12 reversed dedupe pointers,
`successors()` recognizes an exact-content retired-to-active or
retired-to-newer target as the legacy forward edge. It suppresses the inverse
false successor, using active status to disambiguate equal timestamps, and
does not rewrite the historical rows.

Every public `MemoryStore` mutation now passes an early writable-store guard.
Read-only create, update, supersede, delete, pin, apply, touch, and maintenance
paths reject before embedding or another fallible mutation-side effect.

Provider failure, insert failure, or a concurrent loss of the active
precondition therefore publishes no replacement and leaves no newly retired
predecessor.

### Durable pending-to-indexed ingest journal

`TranscriptStore.stage()` normalizes and validates a `Turn` without publishing
it. `append_turn()` is exact-ID idempotent and fails closed on an identity
conflict; the ordinary `append()` API delegates through those two operations.

Schema v13 adds `pending_ingests` and normalized
`ingest_chunk_reservations`. A canonical, text-free manifest records each
turn's complete ordered chunk topology: chunk ID, span, token count, and text
hash. Each member also claims that exact identity in the reservation table,
whose `chunk_id` primary key makes ownership global rather than merely
turn-local. The turn, manifest, and reservations publish in one
`BEGIN IMMEDIATE` transaction.
The receipt then advances monotonically from `pending` to `indexed` inside the
same SQLite transaction that proves every expected dense label, embedding, and
BM25 document length. Completed receipts remain durable, so a later retry with
a different chunker cannot silently replace the topology.

SQLite triggers admit a reservation only when all of its fields equal an exact
JSON manifest member, reject reservation update/delete, reject receipt delete,
and permit only a complete `pending -> indexed` receipt update. Every database
connection enables recursive triggers, so trigger-issued writes cannot skip the
same invariant layer. Supported direct dense and lexical writes may fill only a
pending member. Once a receipt is indexed, a missing or partially retired member
is terminal state: exact retries and caller-supplied lexical rebuild batches
cannot reactivate it. The default no-argument lexical rebuild is narrower: it
derives the live batch from SQLite before clearing postings and may reconstruct
that exact database-selected topology.

Process death or an ordinary index failure after turn publication therefore
leaves an explicit pending owner. `recover_pending_ingests()` reconstructs the
exact chunks from turn spans, re-embeds them, repairs the indexes, and completes
the original receipt. There is no lease or process owner to strand; compatible
writers may finish one another's pending work. The v13 migration seals every
pre-v13 turn that already owns chunks as historical `indexed` state. V12 has no
durable claim that can distinguish interrupted publication from a lexical-only
legacy representation or an intentionally retired all-null chunk; inferring
`pending` could silently resurrect deleted evidence. Legacy turns with no
chunks remain claimable because the migration likewise cannot infer whether
they were intentionally empty or never indexed. In live v13 stores, an exact
retry of an already indexed receipt also skips index addition, preserving any
later intentional chunk retirement.

Before publication, single, batch, and recovery paths take an immutable tuple of
deep chunk copies, pass a separate set of deep copies to the embedder, and
validate that the provider returned exactly one result for every snapshotted
chunk, no extras, omissions, or duplicates, and changed only derived
embedding/lexical fields. The two-copy boundary matters because a provider
cannot mutate nested state in the validation baseline. The finalizer
also rejects any unexpected chunk owned by a manifest turn. Provider output can
therefore neither inject a retrievable row outside the receipt nor complete a
receipt for only a subset of its live topology.

Before embedding and indexing, a batch groups staged chunks by `chunk_id`.
Byte-identical chunks from a repeated explicit turn collapse to one physical
index write; the same ID with different content fails closed. The public batch
result still contains one row for each requested turn, so idempotent inputs do
not become duplicate HNSW labels. Naive explicit timestamps normalize to UTC
before duplicate comparison; omitted timestamps adopt the group's one explicit
time, while repeated all-omitted records reuse the first generated time.

Both `ingest()` and `ingest_many()` reject a read-only database at entry, before
staging, chunking, or embedding. `TranscriptStore.publish_turn(commit=False)`
also requires an active caller transaction, preventing the transaction seam
from publishing a turn without its manifest.

### One durable chunk-index transaction and revision coordinate

Chunk deletion now clears the dense embedding/label, lexical postings and
document length, CAV/head/Hebbian association rows, and consolidation effects
inside one SQLite transaction. `LexicalIndex.delete_chunk(..., commit=False)`
and `AssociationStore.remove_chunk_artifacts(..., commit=False)` are explicit
composition seams for the lifecycle owner.

Only after that durable transaction commits does the process mutate its HNSW
mapping, mark the live label deleted, clear span caches, and invalidate the
source hierarchy. The chunk row itself remains as a provenance owner during
ordinary retirement. If native label retirement is ambiguous, the process
discards the whole local graph and leaves revision reconciliation armed rather
than allowing an unmapped node to occupy a top-k slot.

Chunk addition now stages dense SQLite rows and BM25 postings in the same
transaction. It adds the corresponding HNSW labels before committing that
durable pair; if HNSW, lexical, finalization, or commit-path work fails, SQLite
rolls back and the entire possibly partial local ANN graph is discarded. The
next query reconstructs that disposable graph from authoritative SQLite, so an
unmapped prefix cannot consume top-k slots. An already-present turn can retry
instead of being trapped behind dense-only state.

Both successful addition and deletion advance a durable
`chunk_index_revision` inside their owning SQLite transaction. Live retrievers
compare that coordinate before relying on local mappings, then reconcile the
full authoritative chunk/label set: missing durable rows are added locally,
and mappings absent from or changed in SQLite are retired. This covers
other-process additions, deletions, and equal-count replacements.

Direct lexical add, delete, and rebuild operations advance the same revision,
invalidating source-level aggregates held by another live retriever. Dense
rebuild decodes and dimension-checks every stored vector before publishing a
new label, conditionally installs previously missing labels under a write lock,
and discards the graph on any partial failure.

That discard rule also covers a partially completed cross-process revision sync
and an interrupted native retirement, including exceptions outside the ordinary
`Exception` hierarchy. A process never continues querying a graph after native
state may have diverged from its SQLite label map; the next read reconstructs
the whole disposable ANN graph from durable rows.

Persisted HNSW images now publish through a unique same-directory temporary
file, file flush, and atomic replacement. A failed save preserves the prior
image; a corrupt or torn image is treated as disposable on open and rebuilt
from SQLite, including on a read-only retrieval facade when durable labels are
already present.

### Version-accurate migration baseline

The v11 discourse-snapshot hook now writes `schema_version=11` explicitly,
binding its baseline to the migration that created it instead of to the newest
version known by the running code.

Moving the store through schemas v12 and v13 intentionally invalidates
compiled-cache manifests keyed to an earlier schema. That one-time rebuild is
the safe compatibility behavior; a cache without successor redirects and
pending-ingest receipts is not silently reused under the new contracts.

Read-only behavior is enforced at the facade as well as the stores. After
packing, `build_context()` suppresses memory reheating and live-consolidation
learning when the database is read-only, while returning the same retrieved
context and leaving `consolidation_learned=False`.

### Chunk-scoped provenance validation

When a provenance entry names a chunk, validation now proves all of the
following:

- the chunk exists;
- it belongs to the cited turn;
- its stored text equals the cited turn's exact `[start_char:end_char]` slice;
  and
- the normalized quote occurs inside that chunk, not merely elsewhere in the
  turn.

Supersession validation also requires an active predecessor. The new rejection
reasons distinguish unknown chunks, turn mismatches, span mismatches,
chunk-local quote misses, and invalid memory status.

### Explicit semantic and typed construction identities

The terminal adapter's four semantics retain distinct sealed formats: v2 is
the historical unlinked construction, v3 adds selected-evidence discourse
links, v4 enables post-dedup backfill, and v5 combines links with backfill.
Replay derives both feature flags from the sealed format rather than from an
ambient default.

The reduced terminal CLI exposes those modes explicitly as `v2`, `v3-linked`,
`v4-backfill`, and `v5-linked-backfill`. Omission remains frozen v2. Mode maps
exactly to compilation format; successor artifacts declare their terminal
compilation format, while the v2 top-level projection omits that additive
declaration to preserve its historical bytes and root.

Mode-specific defaults isolate successor roots. In particular, a v4 or v5 run
that retains the legacy default argument cannot target the v2 root or each
other. Full100 construction uses the same routing, and its resumable checkpoint
policy receipts bind the selected mode so resume cannot silently change
semantics. Successor manifest validation checks the selected compilation
format, and resumable construction rejects every all-default root collision
rather than allowing one mode to publish into another mode's lineage.

Typed-additive composition now makes the compatibility boundary equally
explicit:

- `legacy_v1` reproduces dedup-before-lane admission under the original v1
  formats and compact-final provider mode;
- `post_dedup_backfill_v2` performs lane-first admission, exact deduplication,
  authority transfer, and freed-capacity backfill under new v2 identities;
- both sealed historical specialist v2 and v3 remain legacy; repaired
  composition is isolated in `run_reduced_specialist_retrieval_assay_v4.py`
  under format v4, root `reduced-specialist-missing10-v4`, and construction and
  audit names `reduced-specialist-construction-v4.json` and
  `reduced-specialist-target-audit-v4.json`; and
- semantic-binary search remains legacy by default, while its explicit v2 mode
  adds a top-level binding and isolates construction **and audit** formats,
  roots, and filenames.

This is a format and reachability repair, not a new retrieval-result claim.

## Verification evidence

No provider, responder, or judge calls were made, and no historical sealed
artifact was rewritten.

- The final consolidated core lifecycle suite passed **620 tests in 286.93
  seconds** across architecture, compiled cache, condenser, database/schema,
  memory store, validator, transcript store, pending-ingest journal, lexical
  and hybrid retrieval, association store, and consolidation. It includes the
  final reservation-owner, terminal-rebuild, provider-mutation, partial-ANN,
  rollback, read-only, successor, and provenance regressions.
- The final semantic/typed successor-focused aggregate passed **102 tests in
  54.54 seconds**.
- Separate compatibility runs passed 15 adapter/v61 tests and 54 broader
  postseal/answer/judge tests. After a strict fixture correction, the affected
  full100-answer file reran **10/10**. These supporting runs can overlap and are
  not added to the 102-test aggregate.
- The legacy typed golden fixture retained top receipt
  `64404922...a7a3`, and its sampled nested receipts matched commit
  `2124f98^` exactly.
- Before this orphaning repair, the complete matched-eval baseline was 896
  passed and one skipped. That is a prior baseline, not verification of the
  current working tree.

Core and semantic counts are separate and must not be summed into a benchmark
claim. They verify apparatus behavior, not answer quality or 1M-token recall.

## Residual limits

1. **Pending recovery is explicit.** The v13 receipt makes a process-death
   interval visible and replayable, but startup does not automatically run the
   embedder. Until `recover_pending_ingests()` is invoked, pending turns are
   deliberately absent from retrieval rather than falsely reported complete.
2. **The receipt ends at index completion.** With `auto_extract=True`, process
   death after dense/BM25 completion but before derived memory extraction is
   not a pending-index event. The source turn remains intact and an exact retry
   can rerun extraction, but crash-complete extraction would require a separate
   stage receipt rather than overloading index status.
3. **HNSW remains non-transactional but disposable.** Partial in-memory graphs
   are discarded, persisted images publish by atomic replacement, and corrupt
   images rebuild from SQLite. SQLite receipts and revisions, not the native
   image, remain the authority.
4. **Cold derived-state collection is still separate work.** Zero-degree
   Hebbian/consolidation rows and obsolete association artifacts need a bounded,
   receipt-bearing compactor with age and reachability rules. They should not
   be deleted merely because they are currently cold.
5. **Candidate-level mechanism ownership needs fuller receipts.** Future
   receipt work should bind every hydrated, capped, deduplicated, and dropped
   candidate to its producing specialist and final disposition.
6. **Raw SQL remains privileged.** The v13 triggers make supported APIs and
   migrations fail closed, but arbitrary raw SQL can manufacture an initially
   `indexed` receipt because the completeness trigger governs receipt updates.
   This is a documented trust boundary, not a supported recovery or ingest path.

## Next work

First complete the combined regression run. Then expose pending-ingest health
and explicit recovery in operational tooling. In parallel, build a dry-run
cold-state reachability report; only after its owner and successor rules are
auditable should it become a mutating compactor.

Once apparatus verification is clean, construct a newly named v5 terminal
artifact and evaluate it under the existing responder/judge protocol. Until
those calls happen, this repair changes lifecycle correctness and mechanism
reachability only; it does not change the recorded benchmark score.
