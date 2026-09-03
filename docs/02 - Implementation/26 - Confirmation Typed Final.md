# Confirmation typed final

`tools/confirmation_typed_final.py` is the arbitrary-population executable
port of the frozen compact typed-memory final arm. It consumes the exact
in-process confirmation carriers rather than reopening validation artifacts:

- `ConfirmationQueryExpansionContext`
- `VerifiedConfirmationAdaptiveEvidencePlane`
- `VerifiedConfirmationAdaptiveSourceMapPlane`
- `VerifiedConfirmationAdaptiveTailPlane`

Construct `ConfirmationTypedFinalInputs` from those four objects. The
constructor proves that the solver, base map, tail, evidence map, source
population, and context are one lineage and preserve one ordered question
population.

## Provider-free composition

`materialize_confirmation_typed_composition` performs all retrieval-side work.
It groups questions by authenticated namespace and, for each namespace in
source order:

1. revalidates the sealed store bytes;
2. opens `memory.db` once and builds one full-store window index;
3. runs slot closure and active reconstruction for every question in that
   namespace;
4. adapts map, base, tail, full-store, and active evidence independently;
5. performs identity-proven cross-method dedup only after all mechanisms have
   selected;
6. applies non-borrowable lane minima, shared-surplus filling, opaque story
   linking, typed operator validation, and the `COMPACT_FINAL` hard 8k fit;
7. seals the namespace result and releases its index before opening the next
   namespace.

The global closure and composition artifacts preserve original ordinals. Both
seal `maximum_simultaneous_namespace_indexes: 1`; this prevents a confirmation
run with twenty 1M-token stores from retaining twenty full-store indexes at
once. Exact source locators remain only in local audit projections. Provider
messages contain opaque H/G handles and compact typed evidence.

`replay_confirmation_typed_composition` repeats store authentication and the
one-namespace pass and requires the same closure and composition hashes.

## Terra lifecycle

The remaining functions are deliberately separate:

- `publish_confirmation_typed_final_preflight` seals one distinct compact
  Terra prompt per ordered row.
- `approve_confirmation_typed_final_release` authenticates complete native
  request/response journal pairs and authorizes exactly the missing count.
- `run_confirmation_typed_final_provider` is the only function that may build
  a client. It uses `FastCompletionRuntime`, `retries=0`, and refuses orphaned
  requests, foreign checkpoint files, changed release state, or call-count
  drift.
- `materialize_confirmation_typed_final` is client- and store-free. It applies
  the frozen typed completion validator and preserves the adaptive parent
  byte-for-byte for invalid, unsupported, or keep-parent completions.
- `replay_confirmation_typed_final` first revalidates and rebuilds the memory
  composition, then performs a checkpoint-only answer replay. It returns
  `VerifiedConfirmationTypedFinalPlane`, whose ordered `predictions`,
  `result_rows`, and `judge_rows` are the exact downstream specialist seam.

No phase accepts gold or a reference answer. No confirmation treatment data or
real provider was accessed while implementing this stage.

## Verification

`tests/test_confirmation_typed_final.py` covers arbitrary N, two real SQLite
namespaces, the one-live-index bound, compact payloads, source-locator
firewalls, post-selection dedup evidence, exact parent fallback, partial
resume, request-only and foreign checkpoint refusal, resealed schema tamper,
store-free answer materialization, store-revalidating replay, and byte-stable
reruns. The focused plus historical typed/full-store/adaptive regression set is
114 tests.
