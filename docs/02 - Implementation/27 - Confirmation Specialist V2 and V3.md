# Confirmation Specialist V2 and V3

Status: implemented and provider-free verified on 2026-09-03. No confirmation labels, confirmation dataset, or real provider were opened while implementing or testing this stage.

## Purpose

`tools/confirmation_specialist_v3.py` is the arbitrary-population confirmation port of the promoted specialist-final stack. It consumes the exact in-process `VerifiedConfirmationTypedFinalPlane` and `ConfirmationQueryExpansionContext`; it does not rebuild typed composition or load a validation target list.

The stack is additive:

1. V2 construction routes each dated question from its text and typed operator receipt, scans each used immutable namespace once, and adds only applicable numeric, profile/preference, and temporal specialist evidence.
2. V2 answer execution submits only specialist rows. Rows with no applicable or non-empty specialist contribution remain provider-free parent passthroughs.
3. V3 deterministically reconciles every answer in the frozen order: question-bound temporal, sealed numeric, cross-plane parent authority, then V2 fallback.
4. The terminal adapter publishes the resulting rows in `confirmation_terminal_policy_boundary.PARENT_POPULATION_FORMAT`, with a minimal semantic eligibility projection and the exact V3 source-row receipt.

All APIs accept arbitrary positive `N`; no runtime route contains validation ordinals, validation question IDs, target registries, artifact paths, or promoted-population hashes.

## Construction and proof scopes

The construction stage revalidates the immutable stores, groups routed questions by namespace, opens each used database once, and validates the rebuilt cache/window-index receipts against the typed-final closure artifact. It preserves a maximum of one resident namespace index at a time.

Each submitted answer is normally rendered and parsed with `specialist_scoped_completion`. The scoped proof admits only question-local advisory handles and receipt-bound operator evidence. Three legacy proof-topology errors are recognized semantically:

- `numeric group candidates escaped or overlap`
- `numeric operation mode changed`
- `specialist candidate handle map is empty`

For only those shapes, the adapter rerenders the identical sealed provider input through the ordinary typed-final renderer and parser. The transform seals both source and target message hashes. An unrecognized proof or prompt mismatch fails closed; it cannot silently downgrade into the ordinary parser.

## Native Terra lifecycle

The execution sequence is:

```text
publish construction -> replay construction
publish specialist preflight -> approve exact remaining calls
run provider -> materialize V2 -> replay V2
audit V3 lanes -> freeze lane status roots -> materialize V3 -> replay V3
publish terminal parent population
```

The prompt artifact records the complete submitted-versus-passthrough partition. Provider execution uses the shared native Terra checkpoint journal, `retries=0`, unique-prompt accounting, exact-remaining authorization, and request/response pair authentication. Materialization and replay create no client. Passthrough prediction text is copied exactly from the typed parent and consumes no prompt or checkpoint.

Key artifacts under the selected output root are:

- `confirmation-specialist-construction-v2.json`
- `confirmation-specialist-construction-replay-v2.json`
- `confirmation-specialist-prompt-v2.json`
- the shared Terra lifecycle preflight, release, completion, and checkpoint directory
- `confirmation-specialist-answer-v2.json`
- `confirmation-specialist-answer-replay-v2.json`
- `confirmation-specialist-reconciliation-v3.json`
- `confirmation-specialist-reconciliation-replay-v3.json`
- `confirmation-terminal-parent-population-v1.json`

The V3 carrier exposes ordered `predictions`, `result_rows`, `judge_rows`, `status_rows`, and the temporal/numeric/authority lane status-population receipts. Its composition policy binds only the current sealed lane audits and V2 ancestry, rather than the historical 72-call validation population.

## Terminal eligibility seam

`publish_confirmation_terminal_parent_population` is the production adapter from the exact V3, typed-final, query-context, and treatment-preflight objects. `compile_confirmation_terminal_parent_payload` and `publish_confirmation_terminal_parent_sources` expose the narrow deterministic core for testing and integration.

The embedded eligibility input contains semantic projections of `answer_row`, `construction_row`, `prior_answer_row`, and `reconciliation_row`. Population coordinates and bulky proof payloads are excluded. Before publication, the adapter evaluates and replays `SemanticResidualEligibilityDecision`; the terminal loader independently recomputes the same decision from the sealed projection. The parent row separately binds the exact original V3 row receipt, so trimming the gate input does not break ancestry.

## Verification

Focused tests in `tests/test_confirmation_specialist_v3.py` cover question-local route categories, scoped parsing, recognized stale-proof transformation, tamper rejection, arbitrary-`N` mixed submitted/passthrough materialization, exact checkpoint resume, incomplete-journal refusal, V3 frozen precedence, byte-identical replay, and terminal-parent load plus eligibility replay with synthetic policy/treatment/preflight artifacts.

The focused specialist suite passes 7 tests. The relevant historical construction, answer, V3 reconciliation, and scoped-completion regression suite passes 40 tests, including concrete temporal, numeric, authority, and fallback lane precedence.
