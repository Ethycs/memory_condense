# Confirmation query-expansion adapter

This adapter is the executable, population-neutral bridge from the protected
confirmation S0 plane and cumulative stores into the existing matched-eval
query-expansion arm. It preserves that arm's native preflight, runtime identity,
call keys, journal directory, materializer, and replay.

## Dataflow

1. Rebuild and authenticate the protected S0 plane from its sealed policy,
   treatment, cumulative, prompt, completion, and replay inputs.
2. Authenticate every cumulative namespace checkpoint and its self-seals,
   then verify the combined-store receipt against the actual `memory.db` and
   `hnsw_index.bin` bytes.
3. Scan each database read-only and construct a complete
   `FrozenSourceNamespace`, including every source/chunk membership relation.
   Store directories, database/index hashes, namespace order, and the first
   row offset of each arbitrary-size shard remain bound in the context.
4. Reuse `build_query_expansion_population` and
   `preflight_query_expansion`; no prompt renderer is copied.
5. Inspect the native `terra-query-expansion-provider-calls-v2` journals with
   the native runtime. Seal a release containing the exact call plan and the
   number of missing complete request/response pairs. Release itself makes
   zero provider calls.
6. After explicit opt-in and exact remaining-call authorization, delegate
   execution to the native completion runtime. Retries remain zero, foreign
   journals and request-only state fail closed, and resume preserves the same
   runtime provenance and call keys.
7. Load all journals client-free and delegate retrieval/materialization and
   deterministic run/ledger replay to the existing query-expansion functions.

The standard downstream artifacts are therefore the native query preflight,
query run and run replay, runtime ledger and ledger replay. The only additional
artifact is the no-clobber confirmation provider release with its digest
sidecar. This module has no CLI, benchmark reader, gold/reference path,
question allowlist, provider SDK import, or validation population constant.

## Current downstream seam

The query run is one parent input, not a substitute for the rest of the frozen
chain. The direct query-payload parent and V2 evidence-map parent now consume
and authenticate it in their own adapters. Both are required before the
historical source-map step: the validation source-map path consumed the query
run together with the verified V2 map plane, so neither may be silently
skipped. The next unimplemented production stage is the confirmation
source-history/source-map adapter, followed by the existing adaptive/tail,
typed-final, specialist, V3 reconciliation, terminal, numeric-frontier, and
policy-overlay sequence.

The V4 protected-S0 to live answer-plane compatibility bridge is implemented
in the query-payload parent; downstream code must use that authenticated bridge
rather than fabricate a live plane from predictions alone.

## Verification

`tests/test_confirmation_query_expansion_adapter.py` uses only synthetic data
and real temporary SQLite stores. It covers arbitrary counts and shard shapes,
complete source freezing, provider-free native preflight, wrong call-count
refusal, exact native execution and zero-call resume, core materialization and
byte-identical replay, missing-checkpoint and post-freeze index tamper failures,
and absence of validation constants, provider SDK imports, and CLI execution.
