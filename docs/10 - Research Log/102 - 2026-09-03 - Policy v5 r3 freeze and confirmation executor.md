# Policy-v5-r3 freeze and confirmation executor

**Date:** 2026-09-03
**Status:** validation policy frozen at 95/100; confirmation remains unopened

## Result at the freeze boundary

The receipt-bound `policy-v5-r3` validation result is **95/100**. The five
misses are validation ordinals 14, 40, 49, 82, and 94. This is a validation
result, not yet a confirmation/generalization claim.

The implementation was committed as `4c27a5f802dc0537b6eced6eb95939241d7877be`.
Its Git tree is `98c3b373e7a77b5853a7e8a45487dfc007b49ae1`. The separate
freeze-manifest commit is `c72755d`.

The canonical freeze artifact is
`data/policy-v5-r3-confirmation-freeze-v1.json`:

- file SHA-256:
  `1dc9c040962800873f2a1ca2fb57fb4b925f4703fba5f392d60403f1a1586e2b`;
- manifest identity:
  `db17fd410eb5be8b5e6679be4976451af10ea1d74f0ece4fb47fe47db8541259`;
- validation lineage: ten canonical sealed artifacts and six raw Sol journal
  files;
- full100 policy-binding receipt:
  `7cb959a035945d71a0dd33e9f0156bfb7b84c1ede386a5235f43f013b75875a4`;
- confirmation population: 200 rows with ordered-ID root
  `6270b044792dbda79cd79a104ab6a519b2f81980c47522c19a196583d8c0d102`;
- provider calls made by the freeze: zero.

The integrated pre-freeze suite covered the firebreak, question-local
specialist routing, numeric frontier, policy overlay, differential judging,
and persistence repair: **137/137 passed in 149.57 seconds**. The specialist
successor predicate reproduces the same thirteen historical selections while
remaining invariant to renumbering; validation ordinals now act only as
post-selection audit expectations.

## What the frozen policy actually contains

The authoritative 95/100 lineage is not a runtime cascade of every historical
experiment. Its cumulative retrieval parent preserves the ordered S0-S3
ladder, then the successful parent grew through query expansion, source
history mapping, adaptive evidence solving and tail recovery, typed
composition, specialists, residual search, and P/R/L/G terminal composition.
The final overlay arbitrates in this order:

1. supported operator-first numeric proof;
2. accepted typed-final-validator-v5 replacement;
3. byte-exact protected parent.

Earlier fixed-S1 EM, CAV reinjection, and Hebbian/heat arms remain useful
isolated mechanism assays, but they are not imported by the sealed terminal
and policy-v5-r3 chain. Adding them to confirmation now would define a new
treatment rather than reproduce the frozen one. This distinction must stay
explicit in future architecture and score reports.

## Confirmation population and exposure boundary

The validation100 and confirmation200 partitions are disjoint. The source
dataset SHA-256 is
`d6f21ea9d60a0d56f34a05b609c79c88a451d2ae03597821ea3d5a9678c3a442`;
the split-manifest SHA-256 is
`8d5c1885903b199a4ab0859ccabc5ce41d9a105d0c755d3daf33cbfd959995f4`.

Fifteen confirmation answers were potentially exposed in answer-only numeric
metadata. The eventual report must therefore include both the predeclared
full200 result and the non-exposed185 sensitivity result. Neither slice may be
used to retune this frozen policy.

As of this entry, no confirmation treatment has been exported or decoded, no
confirmation prediction exists, gold remains closed, and no Terra or Sol call
has been made for confirmation.

## Streamlined confirmation architecture

The new executor is intentionally a pipeline of small content-addressed
boundaries instead of another copy of the validation apparatus:

1. the firebreak exports only histories, source coordinates, timestamps,
   question text, and question date;
2. the provider-free planner owns arbitrary population and namespace
   scheduling;
3. a namespace workset separates ten owned probes from the complete suffix
   haystack required to reach at least one million tokens;
4. one namespace at a time is ingested into the existing
   `MemoryCondenser`/SQLite/HNSW base store and checkpointed;
5. cumulative S0-S3 retrieval produces compact, monotone, namespace-bound
   evidence receipts;
6. inert Terra prompt preflights reveal exact call counts before release;
7. final predictions are sealed and replayed before any benchmark label can be
   opened; and
8. the evaluator emits exactly 200 question/reference/prediction Sol rows,
   then reports full200 and non-exposed185 scores.

The historical ten-question shard name does not mean “ingest exactly ten
histories.” Each namespace begins at its probe-block boundary and admits
complete histories from that suffix until it reaches at least 1,000,000
tokens. Only the first ten questions belong to the shard. Adjacent suffix
haystacks may therefore overlap. The new workset seals probe membership and
haystack membership separately so this stress condition cannot be weakened by
accident.

## Current implementation boundary

The complete 17-phase production executor is now implemented and synthetic-
tested without confirmation access. It covers namespace ingest, cumulative
S0-S3 retrieval, protected S0, query expansion/direct answer/evidence map,
source streams, adaptive source/evidence/tail stages, typed final, specialist
v3, residual local/global composition, terminal v5, numeric overlay, and the
sealed prediction plane.

Three process boundaries are explicit:

- the standalone exporter is the only process that can open the raw benchmark
  and full freeze, and emits a minimal runtime-policy projection;
- the prediction process has no raw dataset, split, full-freeze, gold,
  reference, judge, or scorer input; and
- the evaluator accepts predictions only through a handoff authenticating the
  run manifest, all 17 checkpoint identities and dependency edges, retained
  provider-journal populations, and the final prediction binding.

The fixed provider-free apparatus suite passes **229/229 in 114.02 seconds**.
The recursive readiness firebreak resolves 319 apparatus files and 303 files
reachable from the prediction entrypoint, with no forbidden evaluator/data
edge, sensitive loader callable, or unresolved dynamic import. Windows
symlink/junction escape tests, interrupted-provider resume, no-clobber seals,
runtime-policy leak rejection, and production-adapter provenance all pass.

At this boundary the confirmation treatment is still unopened and Terra/Sol
confirmation calls remain zero. The only remaining pre-access actions are to
commit this exact apparatus, publish its clean-tree offline-test receipt, and
attest that committed tree. Confirmation export and execution follow that
attestation without further policy changes.
