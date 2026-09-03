# Confirmation query-payload parent

This stage is the first direct-answer parent after confirmation query
expansion. It is population-size neutral and gold-blind.

## Dataflow

1. Rebuild and authenticate the protected S0 plane, S0 prompt, and sealed
   Terra completion.
2. Project that V4 completion into the exact shared
   `VerifiedS0V2AnswerPlane` carrier. The projection has its own canonical
   run/replay and normalized runtime-ledger pair; it makes zero provider calls.
3. Require a complete query-expansion run/replay and runtime-ledger/replay.
   Their seals, snapshot, source population, retrieval, preflight, and run
   lineage are checked before `build_query_fact_population` is called.
4. Call the existing `build_query_payload_answer_plan`. The shared builder now
   accepts only registered V2, V3, or V4 S0 renderers while still requiring the
   parent and source renderer, snapshot, and population identities to match.
5. Publish the authoritative query-payload preflight and a second sealed plane
   containing the exact submitted Terra messages. Fallback-only rows are not
   submitted.
6. Approve a release for exactly the remaining authenticated journal pairs.
   A request without a response, a foreign journal, a changed checkpoint, an
   extra release field, or a changed seal fails before client construction.
7. Delegate provider execution, journal loading, answer materialization, and
   replay to `tools.matched_eval.query_payload_live`. Its runtime fixes retries
   to zero. No second completion runtime or cross-directory journal copy is
   introduced.
8. Replay returns the exact `VerifiedQueryPayloadAnswerPlane` consumed by
   `build_evidence_map_plan`.

## Boundaries

The only private seam is a read-only use of the authoritative query-payload
runtime to authenticate already-complete journal pairs during release. Actual
calls, materialization, and replay stay in the existing public lifecycle.

The implementation does not contain validation constants, sample ordinals,
question-specific branches, benchmark readers, reference answers, or scorer
imports. Query-expansion store access remains upstream; this stage accepts only
the sealed, replayed matched-eval artifacts.

## Verification

`tests/test_confirmation_query_payload_parent.py` covers:

- arbitrary population sizes and exact V4 parent construction;
- protected-parent tamper rejection;
- registered V4 acceptance plus mismatch and unregistered-renderer rejection;
- query-expansion run/replay and ledger authentication;
- exact prompt export and zero-call preflight;
- wrong call authorization, partial-checkpoint resume, and request-only
  refusal;
- exact release-schema rejection;
- provider/materialize/replay accounting; and
- construction of the next evidence-map plan from the returned exact type.
