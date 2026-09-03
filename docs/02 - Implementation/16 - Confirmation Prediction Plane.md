# Confirmation Prediction Plane

`tools/materialize_confirmation_prediction_plane.py` is the provider-free
projection from the final answer-policy output to the evaluator's sealed
prediction plane.

## Input seam

The final confirmation answer compositor is still being assembled, so the
materializer defines the narrow closed source format
`memory-condense-confirmation-final-answer-source-v1`. It has no validation
count, ordinal, or question-ID constants. Its complete ordered rows contain:

- `question_id` and a non-empty, whitespace-canonical `prediction`;
- the prediction's SHA-256;
- the selected upstream source-row receipt SHA-256;
- a self-sealed policy-decision receipt binding the same question and source;
- explicit `fallback_used` and `fallback_reason` fields; and
- a row self-seal.

The source artifact binds the frozen policy, label-free treatment, treatment
preflight, arbitrary population size, and ordered question root, and carries a
top-level self-seal. The eventual final-answer compositor should publish this
format directly or use a small provider-free projection into it.

## Output and replay

Materialization verifies every policy/treatment/preflight seal before accepting
the final source. It rejects missing, duplicate, reordered, empty, noncanonical,
or tampered rows. It then emits exactly
`memory-condense-confirmation-predictions-v1`, whose row schema remains only
`question_id` plus `prediction`, as required by the gold-opening gate.

Fallback provenance is intentionally not added to that downstream closed
schema. It remains authenticated in the source, while the materialization and
replay receipts expose the fallback count and an ordered hash of all fallback
policy-decision receipts. Thus fallback use cannot silently disappear during
projection.

Replay rebuilds the prediction object from the same sealed inputs, compares it
to the source prediction artifact, and publishes byte-identical canonical JSON
and a filename-bearing SHA-256 sidecar. Neither command accepts benchmark,
gold, endpoint, API-key, model, retry, or provider-execution arguments.
