# Cumulative terminal episode closure and packing repair

Date: 2026-08-29

## Result boundary

The sealed cumulative-terminal v1 assay (`589b598a...adeff`) was a useful
negative result, not a release candidate.  A read-only join against the locked
target-owner plan found 20 of 26 required source facts in the exact provider
projection.  The missing facts were two cuisine events, three plant-acquisition
facts, and one Art Cube visit.  The old artifact remains immutable.

The failure was not a larger-context or provider problem.  The complete exact
segments were resident in the local index, but G considered only its prior
attempts plus a small completed/proposed overlay.  Repeated segments from a few
sources then spent the fixed G budget, and the final compact fitter could evict
a fact even after its plane had selected it.

## Generic v2 repair

The v2 compiler treats source and partition equality as receipt-bound linking
edges.  It does not use benchmark ordinals, target source IDs, answer text, or
gold labels.

1. Hydrate exact same-source, same-partition history segments for every G-seeded
   source group from the authenticated resident residual index.
2. Rank opaque partitions by the number of distinct source groups jointly
   supporting question-derived role, entity, action, date, and completed-event
   obligations.
3. Spend the unchanged 24-item/2,400-token G budget as five source-group rounds
   over the top four partitions (20 rows), one authenticated selected anchor
   from each of the top three partitions, and one exact-relation/temporal anchor
   outside the top four.  Oversized rows are still skipped and selection budgets
   remain non-borrowable.
4. Keep semantic disposition separate from `closure_class`, so a completed or
   proposed event remains auditable even when it also entered through episode
   closure.
5. At the final fit, protect the per-plane minima, every post-dedup L
   `packed_novel` row, and every G closure row.  The compiler fails closed if
   this bounded tranche cannot fit; no hard cap is relaxed.

The final retention vector orders authenticated anchor/closure class, absence
of explicit temporal conflict, temporal fit, completed-event support, user
ownership, group obligation support, partition support, and stable upstream
ties.  Opaque source and partition locators remain outside provider bytes.

## Provider-free verification

- Terminal adapter adversaries: 11/11 passed.  They cover the exact 20+3+1
  schedule, outside-cluster escape, plant acquisition versus
  humidifier/shoes/furniture/import noise, January-anchor versus Art Cube
  history, hard-cap retention, exact provenance, per-plane minima, and replay.
- Adapter plus strict wrapper: 18/18 passed.
- Real q82 V6.1 compatibility gate: passed in 282.72 seconds.  Both the Garmin
  fact and the chain/cassette fact are present in provider bytes after the v2
  repair, and deterministic replay is identical.
- New provider calls: zero.

The v2 construction/replay names, policy/selection receipts, route ID, and
default output root are separately versioned.  The next release gate is a new
full exact-11 construction and byte-identical replay, followed by the same
post-seal fact/source coverage audit.  The stale v1 artifact must not be
overwritten or treated as evidence for v2.
