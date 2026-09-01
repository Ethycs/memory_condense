# Namespace index lifecycle and streaming audit

**Date:** 2026-08-28

**Status:** **PROPOSED / NOT IMPLEMENTED**; measured read-only lifecycle audit;
zero provider calls

## Result

The locked typed-final compose path currently retains all ten full-store
window indexes until all 100 question rows have been composed. The retention
is structural, not required by the retrieval result: the closure results keep
their selected candidates, exact local bindings, operator state, and receipts,
but do not retain the window index or namespace cache.

The retaining owner is `_build_full_store_results` in
`tools/run_locked_typed_memory_final_arm.py`. It builds an
`index_by_namespace: dict[str, FullStoreWindowIndex]`, returns that dictionary
to `_composition_projection`, and the latter indexes into it throughout the
100-row composition loop. Each `FullStoreWindowIndex` in turn owns its
`NamespacePartitionCache`, flattened rows, sentence windows, and term, role,
date, and numeric postings.

One frozen offset-000 namespace was measured independently. It contained
7,623 content rows and 55,566 indexed sentence windows. Python live-allocation
measurements were:

| Retained object plane | Incremental live bytes | Approximate MiB |
| --- | ---: | ---: |
| immutable namespace cache | 75,393,396 | 71.9 |
| full window index above that cache | 79,946,548 | 76.2 |
| cache plus full index | 155,339,944 | 148.1 |

Deleting the index while retaining the cache released 80,264,783 live bytes.
The process RSS allocator did not immediately return every released page to
the operating system, so live Python allocations, rather than instantaneous
RSS decline, are the useful lifecycle measure.

Straight ten-namespace extrapolation gives approximately **1.55 GB** for ten
retained cache-plus-index pairs. Retaining ten immutable caches and only one
full index at a time gives approximately **0.83 GB**. The proposed lifecycle
therefore removes about **719 MB**, or **46%**, from this store/index plane.
These are measured single-shard values extrapolated across the ten locked
namespaces, not an observed whole-process peak-RSS claim.

## Why a cache-only handoff is the smallest exact option

The closure artifact must be sealed before final composition. Its SHA-256 is
passed to `adapt_full_store_slot_closure`, becomes part of the full-store typed
contribution, and transitively binds active-reconstruction and final packet
receipts. Composing a namespace before that SHA-256 exists would require a new
receipt protocol or a provisional-result rebind.

The smallest correctness-preserving boundary is therefore a two-phase
in-memory flow:

1. **Closure phase.** For each namespace in frozen population order, read its
   SQLite store once into an immutable `NamespacePartitionCache`, build one
   temporary `FullStoreWindowIndex`, run the ten bound closure questions, keep
   the cache and compact closure results, and release the index before building
   the next namespace's index.
2. **Artifact boundary.** Construct and publish the same ordered cache-receipt
   and closure-question projection. No field or receipt policy changes.
3. **Composition phase.** Rebuild one deterministic index from the retained
   cache for the current namespace, reuse it across that namespace's ten
   question ticks, materialize only plain composition/audit projections, then
   release the active result and index and discard the consumed cache before
   moving to the next namespace.
4. **Final ordering.** Preserve the original question ordinal order in the
   composition payload and require every cache, closure result, and question
   row to be consumed exactly once.

The frozen population was checked directly: it consists of ten contiguous
namespace runs, each containing ten questions at ordinals 0--9, 10--19, and so
on through 90--99, in the same order as the frozen namespace inventory. That
ordering permits the existing ordinal composition loop to remain intact.

This design still performs one physical database read per namespace. The
second index is rebuilt only from already cached immutable rows. On offset
000, two independent builds from the same cache produced byte-equal index
projections and equal index receipts. Repeating all ten closure scans against
the rebuilt index also produced equal provider projections and equal closure
receipts.

## Runtime tradeoff

The cache-only handoff adds one in-memory window-index rebuild per namespace.
Two measured offset-000 builds took 8.412 and 8.318 seconds. Extrapolated over
ten namespaces, the additional compose cost is approximately **83 seconds**.
The corresponding ten-question closure scans took 8.013 and 7.847 seconds and
produced identical results.

Avoiding retention by rereading SQLite in the second phase would lower memory
further, but it would add both a cache read and an index build per namespace,
change the truthful database-read audit from one pass to two, and broaden the
receipt and runtime change. That is not the proposed first refactor.

## Receipt and determinism risks

The proposal is intended to preserve exact output bytes, but the lifecycle
implementation must keep the following invariants explicit:

- cache receipts must remain ordered by the frozen namespace inventory;
- composition questions must remain ordered by the original parent ordinals;
- every rebuilt index receipt must equal the
  `window_index_receipt_sha256` carried by each bound closure result;
- rebuilding an in-memory index must not be reported as another SQLite read;
- process-local lookup hit/build counters must remain prompt-external and must
  not enter a sealed result identity;
- `TypedActiveReconstructionResult` retains its supplied index, so the result
  and its request/hop objects must not escape after their plain projections
  have been materialized; and
- the old index reference must be released before constructing the next one,
  avoiding a transient two-index peak caused by assignment evaluation order.

The active content-derived lookup does not retain a
`FullStoreWindowIndex`; it retains only opaque keys and integer postings under
its separate bounded process cache. The compact closure results likewise do
not pin the cache or index. Those facts are what make the proposed lifetime
boundary possible without changing selected evidence or provenance.

## Required verification before implementation can be accepted

The refactor should not be considered complete without all of the following:

1. a two-namespace lifecycle unit test proving one SQLite/cache read per
   namespace, one shared closure index per namespace, and no returned index
   dictionary;
2. a tracking-index test proving that at most one full window index is live at
   a time in both phases and that every retained cache is consumed once;
3. a fail-closed test in which a rebuilt index receipt differs from its closure
   receipt;
4. exact equality of closure payload, composition rows, selected chunks,
   local provenance, and all associated receipts against the retain-all
   implementation;
5. a provider-free locked-100 replay demonstrating byte-identical closure,
   composition, and preflight artifacts with zero provider calls; and
6. a before/after whole-process benchmark recording peak Python allocations,
   peak RSS, database reads, index-build count, active-lookup build count, and
   total wall time.

## Decision

This audit supports a namespace-scoped index lifecycle as a low-risk memory
refactor. It does **not** record an implementation, a completed streamed
compose, or a new evaluation result. No source file was changed by the audit,
no answer or judge prompt was submitted, and provider calls were **zero**.
