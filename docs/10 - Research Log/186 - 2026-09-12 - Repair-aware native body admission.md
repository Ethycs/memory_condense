# Repair-aware native body admission

**Date:** 2026-09-12
**Status:** real repair admission and exact hydration pass; initial inference running; target open
**Depends on:** [Log 185](185%20-%202026-09-12%20-%20Dense-list%20summary%20repair%20with%20exact%20source%20subdivision.md)

## Change

`tools/assemble_native_spine_admitted.py` assembles original accepted batches and
verified replacement sections into complete native summary bodies. It replays
the repair lineage without provider access, binds replacements to the exact
original rejected response, and preserves every originally valid summary.
Duplicate repair lineages and replacements from another compilation are rejected.

The old assembler required the admitted atom count to equal the original fragment
count. Exact subdivision legitimately increases that count. The successor checks
ordered, gap-free coverage of the complete raw turns, including speaker, hashes,
coordinates and token counts. It reports original fragments covered separately
from additional sections. The raw source bank and all earlier producers remain
unchanged.

Full admission requires every prepared batch to be accepted or have a verified
repair. Explicit `--allow-partial` snapshots contain only complete bodies; a
missing fragment excludes its whole body. A snapshot stays immutable as the live
compiler advances. Its replay retains the original completion status, and a new
snapshot requires a new output root. A partial store cannot be reused as a full
store. Even complete source admission does not assert hierarchy completion,
summary entailment, answer accuracy or a full100 target pass.

`AdmittedSummaryBodies` reads the summary-only SQLite cache under the successor's
own implementation identity. The cache contains summaries and exact raw pointers;
the existing native materializer supplies actual source occurrence identities
and dates before the existing hydrator loads raw evidence.

## Tests

Ten new focused checks pass (`tests/test_native_spine_admission.py`, chunk
`b2b5ec`, 5.61 seconds). The integration fixture uses authenticated completion
journals across the original compiler and all three repair stages. It exercises
subdivision, missing fragments across batch boundaries, complete-body admission,
unchanged original failures, zero-call replay after further compiler progress,
and full completion with more sections than original fragments. Other checks
reject conflicting repair inputs, changed source coverage, roles, hashes,
originally valid strings, excessive summary length and changed database bytes.

Together with the previously recorded 70 focused checks, the total is 80. These
are ingestion and hydration checks, not a fresh answer evaluation.

## Real execution

The existing full Terra worker remains live. A read-only check observed 634
completed batches, 609 accepted batches and 14,246 accepted atoms, with no terminal
handoff result (chunk `23e9d4`). No duplicate source worker was started.

Real assembly uses the complete prepared source population and the final repair
root `eval_results/native-spine-section-refinement-20260912-r1`. Its output root is
`eval_results/native-spine-admitted-body-store-20260912-r1`; explicit partial mode
keeps complete bodies available while initial inference continues. Execution
session `81868` authenticated the full request population before snapshotting
available terminal receipts and exited successfully (chunk `cfc11a`). This
operation has no provider capability.

The snapshot contains 738 originally accepted batches, seven repaired batches,
26 unrepaired batches and 13,041 pending batches. It admits 1,669 complete bodies
with 17,190 sections covering 17,179 original fragments. The eleven additional
sections are the exact subdivisions from the seven repairs. Bodies touching any
missing or unrepaired fragment remain excluded. Both full-source completion and
the full100 target flag remain false.

The persisted store was reopened through `AdmittedSummaryBodies`. All 23 bodies
touched by the seven repaired batches were materialized at one actual occurrence
each, within their own namespaces. The existing summary index and hydrator
returned all 261 expected raw span identities, exact raw text and actual source
dates, with no diagnostics (chunk `786a64`). This check used all compiled summary
terms to exercise hydration and a generous context budget. It used no benchmark
questions or provider calls and does not measure serving latency or accuracy.
The reproducible verification script is
`.tmp/verify_native_admitted_hydration_20260912_r1.py`; its hash is bound in the
verification receipt.

| Artifact under the admitted body-store root | SHA-256 |
| --- | --- |
| `admission-snapshot.json` | `5726c50221665ee15c081e44dad0b730f25d2fc068326ec50aae353f973562e2` |
| `summary-bodies.json` | `a60d1fc4fc4fb618f7e2b9d4c9b50916e9a7aaa12e11dbf55796dfa1970a8456` |
| `summary-bodies.sqlite` | `57a22ee73b1097d4aea4006a26678343b85dc9816282f1dcb2a0144a17f0e9ee` |
| `repair-hydration-verification.json` | `9f0f0b349ef89f2387f39cab0e5d50a3aa641a210308c70326996b71205d08e7` |

The historical full100's 53 implementation files, the live compiler's six files
and the completed repair lineage's eleven files remain byte-identical to their
bound preflights (chunk `827ad8`).

Full inference, the remaining budget repairs, native hierarchy construction and
the fresh joint full100 accuracy/latency comparison remain pending. The earlier
pooled full100 result remains flat 84/100 versus hierarchical 8/100; this ingestion
change supplies no replacement score.
