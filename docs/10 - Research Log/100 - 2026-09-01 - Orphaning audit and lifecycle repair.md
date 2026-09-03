# Orphaning audit and lifecycle repair

**Date:** 2026-09-01

**Status:** implementation complete; combined core and semantic verification
passing; provider-free; no new benchmark score

## Question

Were retrieval failures being amplified by objects or mechanism outputs that
were produced but then lost between their storage owner and their consumer?

## Finding

Yes. The audit separated three cases:

- **data-loss orphaning:** a turn, chunk, or memory crosses a lifecycle boundary
  without a recoverable canonical state;
- **traceability orphaning:** the payload survives but a provenance, successor,
  receipt, or format identity cannot explain how it is reached; and
- **cold retention:** obsolete derived rows remain stored without being part of
  the live evidence path.

The governing rule is now:

> Every producer output must have a durable owner, and every consumer-visible
> identity must resolve to the object, an explicit successor/retirement, or an
> authenticated omission.

## Repairs

- Memory replacement resolves embeddings before mutation and atomically
  retires the active predecessor, inserts a fresh successor, merges exact-
  duplicate provenance, and records every additional duplicate redirect.
- Schema v12 adds `memory_successor_redirects`; `successors()` unions those
  many-to-one edges with the historical scalar correction chain.
- Pre-v12 reversed dedupe pointers remain walkable: exact-content status and
  chronology checks recover loser-to-survivor direction, including active
  equal-time survivors, without rewriting history.
- `create(..., supersedes=...)` can no longer write a ghost or already-retired
  predecessor link.
- Schema v13 adds a durable `pending -> indexed` ingest journal plus normalized
  `ingest_chunk_reservations`. A turn, its canonical text-free manifest, and
  exact reservations publish atomically; the global `chunk_id` key prevents one
  manifest from stealing another manifest's identity. Index completion is
  recorded only in the transaction that proves the exact dense/BM25 topology.
- SQLite triggers require every reservation insert to equal an exact manifest
  member, make receipts and reservations durable, make reservations immutable,
  and admit only the complete monotonic `pending -> indexed` update. Recursive
  triggers are enabled on every connection.
- Interrupted turns remain append-only and explicitly replayable through
  `recover_pending_ingests()`. Completed receipts stay sealed so a different
  chunker cannot silently reinterpret an existing turn.
- Migration conservatively seals every pre-v13 chunked turn `indexed`; without
  a historical pending claim, incomplete-looking rows may be intentional
  lexical-only or retired state and must not be resurrected. Live indexed
  receipts likewise make exact ingest retries index-idempotent after deletion.
- Single, batch, and recovery ingestion retain immutable deep snapshots and pass
  separate deep copies to the embedder. Output must be a one-for-one derivative:
  no missing, extra, duplicate, replaced, or unembedded rows, including through
  nested provider mutation. Finalization independently rejects unexpected
  turn-owned chunks.
- Direct dense/lexical completion accepts pending manifest members only. An
  indexed receipt is terminal and cannot reactivate a missing or retired member.
  Default lexical repair may rebuild only the live batch selected from SQLite;
  a caller-supplied rebuild iterable remains a direct write and is rejected.
- Byte-identical chunks from a repeated explicit turn in one batch collapse to
  one physical embed/index write; a same-ID content conflict is rejected, while
  the caller still receives both idempotent turn result rows.
- Exact repeated explicit records with omitted times reuse the first generated
  timestamp; naive explicit times normalize to UTC before duplicate comparison;
  explicit time or content conflicts still fail closed.
- Read-only `ingest()` and `ingest_many()` now fail at entry, before the
  embedder or any other staged work runs.
- Read-only `MemoryStore` mutation paths likewise reject before embedding.
- Dense retirement, BM25 cleanup, association/Hebbian cleanup, and
  consolidation effects now share one authoritative SQLite transaction;
  disposable HNSW state changes only after commit.
- Dense addition and BM25 publication share one durable transaction. Any
  possibly partial HNSW/addition failure rolls SQLite back and discards the
  whole disposable local graph, preventing unmapped labels from consuming
  top-k slots; an existing-turn retry can heal dense/lexical incompleteness.
- A durable `chunk_index_revision` makes live retrievers reconcile both
  additions and deletions committed by another process, including equal-count
  replacements.
- Direct lexical mutations advance the same revision, so another retriever's
  source cache cannot retain an obsolete partition map. Rebuild validates all
  stored vector shapes before publishing missing labels.
- A partial cross-process ANN sync or interrupted/ambiguous native retirement
  discards the entire local graph; the next read reconstructs it from SQLite.
- HNSW files save through a flushed private image and atomic replacement.
  Failed saves preserve the previous image, and a corrupt image rebuilds from
  authoritative SQLite instead of preventing startup.
- A read-only `build_context()` suppresses post-pack reheating and
  consolidation learning while returning the same evidence.
- Chunk provenance must resolve to the cited turn's exact stored span and
  contain the cited quote.
- The v11 migration baseline now seals schema version 11 explicitly. Schemas
  v12/v13 intentionally cause compiled-cache invalidation/rebuild as their
  successor and ingest-receipt contracts change.
- Reduced and full100 semantic-global construction route explicit v2,
  v3-linked, v4-backfill, and v5-linked-backfill modes to their corresponding
  sealed formats and isolated default roots. Omission remains byte-compatible
  frozen v2; successor manifests validate the format, resumable policy receipts
  bind the mode, and all-default-root collision guards isolate checkpoint
  lineages.
- Typed additive retains exact legacy v1 behavior behind `legacy_v1` and gives
  the lane-first/dedup/backfill successor the distinct
  `post_dedup_backfill_v2` identity. Sealed specialist v2 and v3 remain legacy;
  the repaired path is the new isolated specialist v4 construction/audit under
  `reduced-specialist-missing10-v4`. Semantic-binary default remains legacy,
  while its explicit v2 mode isolates both construction and audit identity,
  roots, and filenames.

## Verification

- The final consolidated core lifecycle suite passed **620 tests in 286.93
  seconds** across architecture, compiled cache, condenser, database/schema,
  memory store, validator, transcript store, pending-ingest journal,
  lexical/hybrid retrieval, association store, and consolidation. It includes
  reservation-owner, terminal-rebuild, provider-mutation, partial-ANN,
  duplicate-label, read-only, successor-chain, rollback, and provenance
  regressions.
- The final semantic/typed successor-focused aggregate passed **102 tests in
  54.54 seconds**. Separate supporting runs passed 15 adapter/v61 and 54
  postseal/answer/judge tests; after strict fixture correction, the affected
  full100-answer file reran 10/10. These supporting runs are not summed with
  the aggregate. The legacy golden top receipt `64404922...a7a3` and sampled
  nested receipts matched `2124f98^`.

These are apparatus checks, not answer scores. The earlier
896-passed/one-skipped matched-eval suite is a pre-repair baseline and is not
used as current-tree verification.

No provider calls, answer calls, judge calls, sealed campaign rewrites, or
score updates were made. Nothing here establishes 95% recall.

## Remaining risk

The v13 journal makes transcript-to-dense/BM25 publication replayable, but
recovery is an explicit operation rather than automatic startup work. Until
`recover_pending_ingests()` runs, a pending turn is intentionally absent from
retrieval instead of being misreported as complete. The receipt currently ends
at index completion: when `auto_extract=True`, a process can still die after
indexing and before derived memory extraction. Source evidence remains intact
and an exact retry can rerun extraction, but a later extension would need a
separate extraction-stage receipt to call the whole ingest workflow crash-
complete.

HNSW remains non-transactional but disposable: partial live graphs are dropped,
persisted images publish atomically, and corrupt images rebuild from SQLite.
The v13 trigger layer protects supported APIs and migrations, which fail closed;
arbitrary raw SQL remains privileged and can manufacture an initially `indexed`
receipt because the completeness trigger governs updates rather than initial
receipt insertion.
Cold zero-degree Hebbian/consolidation and obsolete association state remains a
separate compaction problem. It should first receive a dry-run reachability
report and explicit retention rules. Candidate-level CAV/QK hydration and cap
receipts also need finer mechanism ownership, but neither gap was treated as a
demonstrated data deletion in this pass.

## Next

Finish the combined affected-suite verification, expose pending-ingest health
prominently in operational tooling, and design a receipt-bearing cold-state
collector. Then construct a newly named linked-backfill v5 terminal artifact
and run the existing sealed answer/judge protocol before changing any benchmark
score.
