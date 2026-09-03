# Confirmation source streams

`tools/confirmation_source_streams.py` is the provider-free bridge from the
completed query/evidence-map parents into the adaptive source-map and tail
stages. It reuses the existing matched-eval implementations rather than
creating a confirmation-only retrieval algorithm.

## Executable contract

`materialize_confirmation_source_streams(...)` accepts:

- an authenticated arbitrary-size `ConfirmationQueryExpansionContext`;
- the exact completed query-expansion run/replay/runtime artifacts;
- the exact direct-query-backed `EvidenceMapPlan`; and
- its replayed `VerifiedEvidenceMapPlane`.

It produces one `ConfirmationSourceStreamsResult` with:

- provider-free query-guided scan and query-expansion-repack-v2 artifacts;
- a sealed question-only partition eligibility manifest;
- a replayable `PartitionScanV2Generation` built from read-only stores;
- the V2 map adapter under consolidated obligations and state-chain direct
  authority;
- exact D/P/G `VerifiedLockedSourceGateRow` values;
- the frozen `d1-p0-g1` base population using the admitted V1 direct stream;
  and
- a second population using the repack-v2 direct stream for adaptive tail.

`replay_confirmation_source_streams(...)` rebuilds every provider-free child,
requires byte-identical run/runtime/eligibility/partition/plane seals, and
returns the same typed in-process result. The entry point has no provider,
gold, judge, validation ordinal, or fixed-population argument.

## Retrieval and selection rules

Partition eligibility is exactly:

```text
route.modifiers.requires_temporal_metadata
OR route.modifiers.requires_complete_frontier
```

The route sees only the dated question. Partition generation groups questions
by authenticated namespace and opens each verified `memory.db` read-only once.
Store database and index bytes are checked before and after construction.

Each D/P/G method keeps its own selected span order. Exact protected-S0
duplicates are excluded only after bounded selection by the existing guided,
repack, and partition implementations; source IDs then collapse to first
distinct order within that method. Cross-method overlap remains visible until
the later source-gate mapping cache, preserving method credit.

## Sealed artifacts

The output root contains:

- `confirmation-source-streams-v1.json`;
- `partition-eligibility-v9.json`;
- `partition-scan-v2-generation.json`;
- `query-guided-scan-v1/`; and
- `query-expansion-repack-v2/`.

Replay adds same-byte replay artifacts. The top plane binds all child hashes,
the map-adapter receipt, both source-population receipts, the frozen policy
receipt, population size, and activation count.

## Verification

`tests/test_confirmation_source_streams.py` covers an arbitrary non-100
population backed by real SQLite stores, full materialize/replay, the exact
route union, selection-before-dedup, state-chain authority-profile selection,
zero new provider calls, post-freeze store tamper, and resealed component
tamper. The focused test plus guided, repack, partition-v2, map-adapter,
guided-payload, and locked-source-gate regressions pass together.
