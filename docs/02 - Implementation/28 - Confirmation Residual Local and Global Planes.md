# Confirmation Residual, Local, and Global Planes

Status: implemented and provider-free verified on 2026-09-03. The implementation did not open confirmation labels, reference answers, or a provider/model endpoint.

## Purpose

`tools/confirmation_semantic_planes.py` is the arbitrary-population continuation from the authenticated specialist-V3 parent into the frozen semantic stack:

```text
question-local eligibility
  -> R7 semantic residual search
  -> V6.1 source-group and episode reinjection
  -> V7 semantic global completion
  -> linked + post-dedup-backfilled terminal-v5 compiler
```

The stage consumes `ConfirmationTerminalInputs`, `VerifiedNamespaceStoreSet`, a sealed supplemental semantic-vector release, and a narrow `ProtectedV3EvidenceAdapter`. It emits per-namespace checkpoints, typed in-process result rows, one deterministic merged materialization, and the exact plan export accepted by `compile_confirmation_terminal_v5_plan_export`.

No mechanism call receives a validation ordinal, target list, label, gold answer, population count constant, or provider object. The historical terminal-v5 consumer still authenticates an `ordinal` identity field. The adapter adds that field only after eligibility, all retrieval, evidence selection, post-selection deduplication, and terminal compilation are complete.

## Frozen policies

Construction instantiates the existing core policy classes; it does not copy their algorithms:

- `SemanticResidualEligibilityPolicy()` — receipt `ebaa6acade2c12f2dcf5f5e52e8e45661870ba916e85fb0f4a79bab5c2ccc955`.
- `SemanticResidualPolicy(max_cell_tokens=2048, payload_token_cap=2400, cosine_upper_bound_floor=.05, specificity_upper_bound_ratio=.75, dual_gate_enabled=True, classifier_mode="evidence-conserving-fail-open")` — receipt `288c9a08051f547626836a3b08fcc85a0844fe1397b27a3f6a978b8801ee6e88`.
- Frozen V6.1 `SourceGroupReinjectionPolicy()` — receipt `c15f430054445dc96c246a3ba156710a6990b807c07dea571f042104a52795df`.
- Frozen V7 `SemanticGlobalCompletionPolicy()` — receipt `504cab6a3d145442e7ebc9d1efa71ac9673249c01092e40cec5f00837157bb61`.
- `SemanticGlobalTerminalPolicy()` with selected-evidence discourse links and post-dedup backfill enabled — receipt `e2b3b5a5eb9dabf4841b56e30ab60998fd12bbfb552a2cff76a594e90f196d3b`.

The runtime compares both every projection in policy-v5-r3 and every compiled receipt above. Drift fails before a namespace is opened.

## Supplemental BGE facet lifecycle

The staged S0–S3 configuration intentionally freezes only raw and dated queries. R7 needs every string returned by `semantic_residual_query_facets(dated_question)`. Re-embedding those facets after Qwen loads would violate the one-model-at-a-time residency boundary.

`prepare_confirmation_semantic_facet_vectors` is therefore a separate phase-A freezer. It receives the completed `StagedPreparationExecution` and the same `StagedPreparationBackend`, derives every facet from question text only, and calls `freeze_query_batch` while the original BGE/embedder and embedding identity remain resident. It publishes one vector artifact and checkpoint per namespace plus a population manifest. The original S0–S3 `FrozenQueryDescriptor` is referenced by SHA-256 and remains byte-for-byte unchanged.

`execute_staged_confirmation_cumulative` exposes the safe point as the optional typed hook:

```python
before_bge_release(preparation, preparation_backend) -> None
```

The hook runs after all namespace preparations and before `release_bge()`. A coordinator closes over the cumulative inputs and output root and invokes the facet freezer there. Once BGE is closed, `publish_confirmation_semantic_facet_release` binds the supplemental preparation to the exact BGE release receipt. The later loader requires that same receipt inside the authenticated Qwen barrier.

Existing vector/checkpoint pairs are fully verified and reused without an embedding call. A partial pair, changed sidecar, different embedding identity, changed staged descriptor, changed question/facet order, or invocation after BGE release fails closed. `load_confirmation_semantic_facet_vectors` opens only one namespace artifact at a time, so the semantic phase never retains the full population of vectors.

## Namespace execution

`ProductionSemanticNamespaceBackend.open_namespace` verifies the staged combined-store bytes and opens one namespace database read-only exactly once for its eligible rows. During that scope it:

1. streams the complete discourse source population;
2. reconstructs the frozen namespace and partition cache;
3. builds the exact full-store window index;
4. loads exact stored chunk embeddings and builds the shared semantic residual index; and
5. authenticates exactly one populated fixed-interval episode artifact and the frozen episode policy.

Namespaces containing only ineligible rows receive a checkpoint but are not opened. All eligible questions in a namespace reuse the same resident index, stored vector set, and episode store. The objects are discarded and the database is closed before moving to the next namespace.

## Evidence and terminal composition

`SpecialistV3ProtectedEvidenceAdapter` joins the current terminal parent back to exact specialist-V3, specialist-V2 construction, and typed-final composition ancestry through the sealed source-row receipt. It delegates P-plane reconstruction to the existing `_protected_evidence` path, which in turn uses protected-parent contribution rehydration and visible specialist local evidence. It does not fabricate provider P rows.

For each eligible question, the implementation calls and immediately replays the existing residual, reinjection, global, and terminal functions. R selects against the full index before protected-evidence deduplication. V6.1 reconstructs authenticated opaque source groups from the exact R/P owners. The cumulative P/R/L binding union is checked for unique exact spans only after those selections. V7 then selects globally and excludes the protected union after selection. This preserves the evidence-conserving rule: protected evidence may suppress a duplicate output, but it may not suppress a candidate before the mechanism has considered it.

The terminal adapter uses the exact protected-owner rows emitted by the R7 terminal renderer, not a synthetic substitute. It compiles with `enable_selected_evidence_discourse_links=True` and `enable_post_dedup_backfill=True`, replays the typed `SemanticGlobalTerminalCompilation`, builds the frozen answer-plan core, adds downstream identity fields, and passes the resulting question assay through the existing terminal-v5 validator. `publish_confirmation_terminal_v5_plan_export` is therefore the authority for the final provider plan; this module does not re-encode provider messages.

`ConfirmationSemanticQuestionRow` retains the exact typed `SemanticResidualSearchResult`, `SourceGroupReinjectionResult`, `SemanticGlobalCompletionResult`, and `SemanticGlobalTerminalCompilation` for eligible rows. All are absent for ineligible rows. Every durable artifact asserts zero new/physical provider calls and zero retained transformer token state.

## Checkpoint and replay behavior

Each namespace checkpoint binds the store identity, supplemental vector artifact, protected adapter, frozen policy receipts, ordered parent receipts, eligibility decisions, and exact mechanism projections. Publication is no-clobber. A resume recomputes from authenticated stores and reuses a checkpoint only when its canonical bytes match. Callers may additionally provide the expected checkpoint SHA-256 keyed by namespace receipt.

`replay_confirmation_semantic_planes` reconstructs the typed results, requires every externally sealed namespace checkpoint and merged materialization, and publishes a byte-identical replay. Tampered bytes or sidecars, changed namespace membership, missing facets, policy drift, store drift, or ancestor drift fail closed.

## Public integration sequence

1. Run staged preparation with a `before_bge_release` closure that calls `prepare_confirmation_semantic_facet_vectors`.
2. Bind that preparation using `publish_confirmation_semantic_facet_release` and the exact release receipt later present in the staged barrier.
3. Load the bounded vector inventory with `load_confirmation_semantic_vector_release`.
4. Construct `SpecialistV3ProtectedEvidenceAdapter(v3_plane, typed_plane)`.
5. Call `materialize_confirmation_semantic_planes`; pass its `terminal_plan_export` directly to the existing terminal policy/preflight lifecycle.
6. Call `replay_confirmation_semantic_planes` with the sealed materialization and checkpoint hashes before provider authorization.

## Verification

Focused coverage is in `tests/test_confirmation_semantic_planes.py` and the staged tests in `tests/test_confirmation_cumulative_retrieval.py`. It checks the exact five policy receipts, population/label-routing rejection, eligible/ineligible mixing, exact typed results, one namespace open, provider-free materialization, checkpoint resume, byte-identical replay, tamper refusal, facet cache verification without re-embedding, and event order `prepare -> facet freeze -> BGE close -> Qwen load`.
