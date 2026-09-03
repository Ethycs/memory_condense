# Confirmation evidence-map parent

This stage carries the frozen confirmation chain from the replayed direct
query-payload answer into the V2 evidence map. It is population-size neutral,
gold-blind, and deliberately stops before the historical standalone evidence
solver.

## Dataflow

1. Accept an exact `ConfirmationQueryPayloadPlan` and the exact
   `VerifiedQueryPayloadAnswerPlane` returned by its client-free replay.
2. Call the existing `build_evidence_map_plan`. The authoritative builder
   verifies the direct-answer run, runtime ledger, adapter population,
   retrieval, snapshot, row order, route, aliases, and retained query delta.
3. Call the existing `preflight_evidence_map` and separately seal a
   confirmation prompt plane. That plane contains exactly the native
   `provider_prompts` for submitted rows; state-chain preservation rows remain
   unsubmitted.
4. Inspect the native `terra-query-evidence-map-v2-calls` namespace. Every
   request must have a response, every journal must authenticate against the
   native runtime identity, and foreign state is refused.
5. Seal a provider release for exactly
   `required unique prompts - authenticated complete journals`. The release
   records the completed journal receipts and fixes retries to zero through
   the native runtime contract.
6. After rechecking the release and checkpoint snapshot, construct a client
   only when calls remain. Execution is delegated to
   `run_sealed_two_pass_provider`; resume still uses the native total prompt
   population so its runtime provenance is unchanged, while the confirmation
   boundary requires authorization for only the missing physical calls.
7. Load completed journals client-free, delegate parsing and materialization
   to `materialize_evidence_map`, and delegate terminal replay to
   `replay_evidence_map`.
8. Return the exact `VerifiedEvidenceMapPlane` expected by the downstream
   source-map adapter.

## Boundaries

The adapter does not load benchmark samples, labels, references, validation
artifacts, fixed ordinals, or question-specific policy. It makes no calls in
planning, prompt export, release, materialization, or replay. Only the explicit
provider method can construct a client, and it does so after all seals and the
exact remaining-call authorization verify.

One private native-runtime seam is used read-only during release to validate
already-complete checkpoint pairs. Provider execution, completion provenance,
map parsing, evidence citation validation, runtime-ledger construction, and
replay all remain in `query_evidence_map_solver_v2_live`.

The evidence-map pass is retained because it changes representation: it emits
individually cited candidate facts and locally salvages valid items. The
rejected standalone solver is not invoked here; later frozen stages consume
the verified map plane through their original source-map and adaptive-policy
path.

## Verification

`tests/test_confirmation_evidence_map_parent.py` covers:

- a full synthetic query-payload to native evidence-map lifecycle;
- byte-for-byte equality between exported prompts and the authoritative
  preflight prompts;
- exact wrong-call refusal before provider construction;
- partial-checkpoint resume with one hit and only the remaining call released;
- request-only checkpoint refusal;
- a resealed release-schema extension rejected before client construction;
- direct-parent binding tamper rejection; and
- return of an exact replayed `VerifiedEvidenceMapPlane`.

The focused authoritative V2 evidence-map suite is also run unchanged to guard
against adapter-induced regressions.
