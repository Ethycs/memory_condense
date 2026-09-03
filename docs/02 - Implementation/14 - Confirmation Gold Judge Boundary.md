# Confirmation Gold Judge Boundary

`tools/confirmation_gold_judge_scaffold.py` is the evaluator-only boundary
between sealed confirmation predictions and benchmark gold. It contains no
provider client and cannot execute Sol calls.

## Required order

1. Verify the externally pinned, sidecar-sealed frozen-policy manifest.
   Verification covers the closed freeze schema, confirmation-candidate status,
   manifest self-identity, treatment-policy self-identity, report-only
   validation result, and the explicit prohibition on validation-result runtime
   use. An arbitrary JSON file is not a policy freeze even if its SHA-256 is
   supplied by the caller.
2. Verify the closed, label-free confirmation treatment and deterministically
   replay its provider-free pipeline preflight.
3. Verify an externally pinned, canonical prediction handoff with status
   `predictions_sealed_evaluation_unopened`. The evaluator derives the run root
   from this handoff; it has no independent prediction-path argument.
4. From that handoff, authenticate the fixed-name run manifest and its closed
   v5-r3 schema, source-policy/treatment/preflight/population bindings, sanitized
   runtime-policy digest, 1M-token workload, exact 17-node production DAG,
   the exact ordered production-adapter identity SHA-256 for every phase,
   retry-zero runtime, and gold-blind safety declaration.
5. Authenticate all 17 fixed-name, sidecar-sealed phase checkpoints in order.
   Every checkpoint must bind the run manifest, all prior checkpoint digests,
   its phase's manifest-bound production-adapter identity, exact provider
   requirement/accounting receipts, population size, artifact bytes, artifact
   sidecars, and its own identity. The handoff's aggregate Terra accounting and
   final-checkpoint digest must replay exactly from this chain.
6. Accept the complete prediction artifact only through the final
   `sealed_predictions` artifact binding shared byte-for-byte by the final
   checkpoint and handoff. It must contain exactly one non-empty prediction for
   every treatment question in the locked order and bind the source policy,
   treatment, and preflight.
7. Only after all provenance checks succeed, open the evaluator dataset and split,
   reconstruct the confirmation population, and compare its ordered IDs,
   normalized bindings, raw-record bindings, and complete treatment projection.
8. Publish a canonical, no-clobber judge plan containing exactly `N`
   question/reference/sealed-prediction rows. The plan reports exactly `N` Sol
   calls that an external executor would make, zero physical calls by the
   scaffold, and no available execution path.
9. Score a separately sealed external verdict plane. Report both the full
   population and the subset excluding identities recorded in the answer
   metadata exposure audit.

For the production confirmation population, the locked exposure expectation is
15 of 200 identities, so the sensitivity denominator is 185. The code does not
encode 200, 185, question IDs, or ordinals: all counts and ordered roots come
from sealed inputs and are checked at runtime.

## Boundary assumptions

- The policy manifest is intentionally opaque to this adapter. Its exact bytes
  outside the freeze contract are authenticated by the caller-provided SHA-256
  and filename-bearing sidecar; predictions must bind that same digest. Its
  confirmation static root (dataset, split, count, ordered IDs, normalized
  bindings, and raw-record bindings) must equal the treatment exactly.
- The label-free treatment and preflight must also have filename-bearing
  sidecars. The treatment is independently replayed through the existing
  confirmation preflight compiler, so matching sidecars alone are insufficient.
- A standalone prediction file is deliberately insufficient. The judge CLI
  exposes only `--prediction-handoff` and
  `--expected-prediction-handoff-sha256`; the prediction path and digest come
  solely from the authenticated final checkpoint/handoff binding.
- Copying or resealing one artifact cannot complete the gate: the run manifest,
  its ordered production-adapter identities, all checkpoint dependency roots,
  per-phase artifact bindings, aggregate provider accounting, and handoff
  identity must agree simultaneously.
- Dataset and split identities come from the treatment seal. Gold is never
  opened to select, repair, route, or fill predictions.
- The exposure audit is externally pinned. Its answer values are never copied
  into the judge plan or score report, and per-row exposure membership is not
  sent to the judge.
- The emitted judge plan is gold-bearing evaluator material. It must not be
  returned to retrieval or answer-generation stages.
- An external, separately authorized executor is responsible for producing the
  complete `memory-condense-confirmation-sol-judge-results-v1` verdict artifact.
  This scaffold only validates and scores it.

## CLI surfaces

`compile-plan` publishes the inert `N`-row judge plan. It requires the sealed
prediction handoff and has no `--predictions` option. `score` publishes the
aggregate score report. Neither command accepts endpoint, model, API key,
retry, or provider-execution arguments.
