# Confirmation adaptive evidence and tail

**Status:** executable, provider-isolated, arbitrary-size adapter complete; no
confirmation records or live provider endpoints were opened during construction.

`tools/confirmation_adaptive_tail.py` ports the two post-source-map planes used
by the frozen `policy-v5-r3` lineage. It does not introduce a new retrieval
policy:

1. The adaptive evidence solver combines the replayed V2 evidence-map plane
   with post-map fact unions from the base source mapper. Rows with no admitted
   source fact are preserved as exact parent no-ops and consume no Terra call.
2. The source tail examines the same base mapper result. It advances one
   method-local source only when the row remains unresolved and the base union
   retained zero facts. Rows already carrying a fact remain pending for the
   solver; satisfied or exhausted rows remain explicit decisions.

## Frozen policy bindings

The adapter requires both exact source populations emitted by
`ConfirmationSourceStreamsResult`:

- base direct profile `query-admitted-delta-v1`;
- deep direct profile `query-repack-selected-before-dedup-v2`;
- base lane budgets D1/P0/G1;
- consolidated obligations;
- state-chain direct authority; and
- the historical route-specific tail lane order and direct-repack threshold.

Logical tail selection occurs before physical mapper-work deduplication. Exact
fact validation and post-map union deduplication occur only after the selected
source is mapped. Thus an EM/source fact is never removed before selection,
while a fact duplicating protected direct evidence is excluded afterward and
remains represented by the existing direct-evidence authority.

## Typed pipeline seam

`confirmation_adaptive_upstream(...)` joins four exact, replayed parents in
memory:

- `ConfirmationSourceStreamsResult`;
- `VerifiedConfirmationAdaptiveSourceMapPlane`;
- `EvidenceMapPlan`; and
- `VerifiedEvidenceMapPlane`.

It verifies source-population, query-map, evidence-map row, plan-row, and replay
bindings without reopening the benchmark. The downstream outputs are
`VerifiedConfirmationAdaptiveEvidencePlane` and
`VerifiedConfirmationAdaptiveTailPlane`. Both retain their exact plan object
and sealed preflight, release, run, and replay artifacts. The tail plane also
exposes exact `FastMaterializationQuestionPlan`, `SourceMapperMaterialization`,
`TailFactUnionRow`, and all-row decision tuples for typed composition.

## Provider lifecycle

Solver and tail calls use separate native `FastCompletionRuntime` checkpoint
namespaces. Each lifecycle is:

```text
typed planning
  -> sealed exact prompt preflight
  -> authenticated request/response checkpoint scan
  -> release for exactly the remaining calls
  -> provider-run (retries = 0)
  -> client=None materialization
  -> byte-identical replay
```

A response-less request, symlink, foreign checkpoint file, changed parent,
extra release field, call-count mismatch, or incomplete completion population
fails closed. The client factory is unreachable from planning, approval,
materialization, and replay. An empty solver or tail population is an explicit
zero-call lifecycle rather than an error.

## Verification

`tests/test_confirmation_adaptive_tail.py` covers arbitrary 1- and 3-row tail
populations, exact no-op parent preservation, fake-client solver execution,
partial tail resume, request-only and foreign-state rejection, parent/release
tamper rejection, and typed byte replay. The focused plus adjacent source-map,
source-stream, and tail-typed suite passes 26 tests.
