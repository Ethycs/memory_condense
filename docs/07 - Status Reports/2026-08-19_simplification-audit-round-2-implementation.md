# Simplification audit round 2 — implementation report

**Date**: 2026-08-19, same day as the audit
**Base**: `main` at `8aa47c8`, all changes uncommitted in the working tree
**Scope**: every finding in `2026-08-19_simplification-audit-round-2.md`, implemented by parallel agents in five conflict-free waves (Phase A correctness → Phase C dead code + transcription tables → persistence/modeling/receipts → round-robin/wrappers/grab-bag → Phase D structural), with the full test suite run green between waves.

## Bottom line

- **107 tracked files changed, +5,784/−6,785; 8 new shared modules (+458)** → net **≈ −540 lines** while adding docstring-carrying shared infrastructure and 7 new unit tests.
- **Full suite green at every checkpoint**: baseline 1854 passed → final run after Phase D (see terminal). No digest, receipt, or identity drift anywhere — every digest-bearing refactor carries a programmatic before/after proof.
- All 23 findings plus the dead-code and structural sections are done, except items the audit itself marked as author decisions (listed below).

## Real bugs fixed (deliberate behavior changes)

1. **Per-source fairness in packing** (`expansion_ordering`): queue grouping skipped `turn.source_id`, making documented per-*source* fairness per-*turn* for turn-hydrated results. Now groups by `_result_source_id`. (A2)
2. **Codex-route 400** (`eval/responder.py`): `temperature` was passed unconditionally; now guarded via the shared `build_completion_request`. (A5)
3. **Judge crash on None content** (`eval/judge.py`): unguarded `.strip()` AttributeError → now an explicit FAIL verdict with a parse-failure reason. (A5)
4. **Recall CSV drift** (`offline_modes`): now `DictWriter` over `QuestionRecall.model_dump()` — the CSV gains the two silently-missing columns (`evidence_source_hit`, `survives_horizon`, → 126 columns); all 124 pre-existing columns keep byte-identical values and order; the legacy `in_header` alias is kept for the frozen-v3 merge tools. (A6)
5. **Silent NaN in digests** (`compiled_cache`, `transition_trace`): canonical-JSON digesting now uses the domain encoder with `allow_nan=False`; NaN raises instead of serializing. Verified no real input or pinned digest is affected. (B18)
6. **Missing-witness hazard** (closure engine): all seven store probes route through `_probe(...)`, which records the witness in both branches — a forgotten witness is now structurally impossible (was silently degrading `completion()`'s claims). (B21)
7. **Migration version publication** (`db.py`): published inside the DDL transaction by the runner; the 10 hand-written tails (fail-silent re-apply hazard) are gone. (A8)
8. **sha validators**: 9 copies → domain validator (lowercase-normalize). Acceptance-only widening; hexdigest output is always lowercase. (A9)

Minor user-visible wording: benchmark/cache invalid stress-flag errors now say `--stress-question-offset must be non-negative` (flag-named) instead of internal wording.

## Notable non-changes (divergences the audit flagged that turned out intended, and were preserved)

- **Stress composition** (A10): offline_modes' deferred sharding is the *documented* behavior (canonical ten-question pool = causal cache identity); unifying on benchmark/cache would have changed `sample_sha256`. All three modes now share `runtime.prepare_samples` with the divergence as one explicit parameter; byte-identical samples proven per mode.
- **mem0 fingerprint encoder** (B18): kept `default=str`/`ensure_ascii=True` — it feeds the versioned `mem0-oss-2.0.18-certified-local-v1` protocol string; unification needs a protocol version bump.
- **`runner._prompt_tokens`** (B22): its twin was deleted, but converting to `count_chat_prompt_token_proxy` would add framing tokens and change reported numbers; left as the sole copy.
- **Frozen-v3 / firebreak tools**: keep self-contained `file_sha256` copies — the bootstrap preflights against an archived source tree without `domain/`.
- **`direct_tokens` denominators** (A11): parameterized, and found to be *textually* divergent only — `bounded_anchors` is already capped, so the arms currently behave identically.

## New shared modules

`domain/integrity.py` (canonical `file_sha256`), `domain/text_numbers.py`, `domain/ranking.py` grew `weighted_fair_order` / `round_robin_unique` / `source_rows_with_fallback` / `min_max_normalize` / `softmax`, `associations/expansion_guards.py`, `eval/_completion.py`, `eval/_identity.py`, `eval/runtime.py` (`prepare_samples`, `transient_runtime_controls`, `run_provenance`), `eval/search_kwargs.py`, `modeling/checkpoint_identity.py`.

## Digest/identity stability proofs performed

- Receipt sealing mixin (B12): 21 recorded instances across all 15 classes, 0 digest diffs; canary `eval_results` receipts covered.
- policy_gate identity (B13): 22 serialized identities byte-identical (JSON string compare incl. key order).
- cli_config (B14): 21 arg-namespace scenarios + 6 error paths + attribute-level diff of all 152 parser actions — identical.
- Checkpoint manifests (B17): format string parameterized so every pinned checkpoint digest is unchanged.
- Round-robin (B19): 3,000-trial randomized differential test against verbatim originals.
- campaign_merge (D3): differential test vs HEAD implementation, outputs and error strings byte-identical.
- qwen linker identity (D4): payloads and `linker_identity_sha256` byte-identical, strict-mode errors preserved.
- Snapshot rows (B16), search kwargs (A7), sample composition (A10): direct before/after byte comparisons.

## Structural (Phase D)

- `_build_expansions`: 707 lines → 35-line orchestrator over six phase methods + `_ExpansionPass` state (+ `_trace_row` factory; 8 mutation sites now routed through the state object).
- `closure/engine._walk`: 428 lines → `_ClosureWalk` owner (scope/seed/expand/receipts); engine.py at 1,258 lines, 42 under the tripwire.
- `merge_benchmark_reports`: 12 named stage validators + `_MergeAccumulator` (11 accumulators); +394 lines traded for testability, 7 new stage unit tests.
- Representative retrieval: shared plan kwargs (13, not 15), one `qwen_linker_identity(strict=)`, span ordering routed through `evidence_span_sort_key` everywhere.

## Open author decisions (deliberately not decided here)

1. **`"causes"` in compiled `TEST_RESULT_RELATIONS`** — compiler now imports the semantics frozensets but explicitly subtracts `"causes"` to preserve behavior (`compiler.py`, commented). Decide: should compiled obligations credit it?
2. **Episode source-conflict policy** — detection is shared (`RetrievalResult.source_hints`), but the three responses (drop fail-open / raise / silent skip) are preserved and commented at each site. Pick one policy.
3. **Residual divergent source keys** — `heat_diffusion._default_source_key` (turn-first), `span_source_queries` :525/:558 (turn-only), and the fourth shim `query_routing._retrieval_source_id`, all left as-is.
4. **Now test-only** (deletion deferred as author calls): `expand_hebbian`, `observe_retrieval_access`, `HeadKVStore.indices_for_episode_ids`, `HeadAssociationGraph.prune_neighbors`.
5. **`run_diffuse_treatment_sample` deleted** (92 lines, zero callers/tests) — recover from git if it was meant to be wired up.
6. **D5 not done** (audit marked it author decision): `INICoverageSelector` accepting the three `active_partition_*` kwargs to satisfy its own Protocol, which would delete the `inspect.signature` negotiation in `expansion_assembly.py` and its partial-capability test doubles.
7. **Injected trace rows** omit `selector_output_rejection` while original rows always carry it — preserved via an explicit `_OMIT` sentinel in `_trace_row`; arguably worth unifying.
8. **mem0 canonical-JSON unification** — needs a certified-protocol version bump (see above).
9. **`direct_tokens` denominator** — unify the textual divergence or keep the parameter.

## Follow-on: dataclass validation cleanup (same day)

After the audit landed, the contract dataclasses were brought onto one declarative validation idiom, `domain/discourse.py` first and then the second-tier model files (`search/episodes/*`, `domain/discourse_routing.py`, `associations/association_models.py`, `application/discourse_sources.py`, and the eval diffuse models):

- `domain/_discourse_identity.py` gained the shared micro-validators (`_nonnegative`, `_positive`, `_finite`, `_strict_int`, `_choice`, `_labeled`, `_as_tuple`, `_unique_nonempty`, `_sorted_unique`, `_sorted_unique_nonempty`); every hand-rolled `object.__setattr__` rebind that was expressible became a `normalize_fields(...)` call.
- The monster `__post_init__` bodies (`ClosurePlan`, `ClosureReceipt`, `EpisodeRepresentativeRetrievalPlan`, `AttentionHeadSurpriseReceipt`) now read as named validation phases; the re-typed payload projections in `ClosurePlan.identity_payload` were replaced with reflective `identity_payload()` methods on the child classes.
- Everything not byte-identically expressible stayed explicit, with comments where a check is deliberately stricter than the shared validator (reject-uppercase digests, verbatim-stored IDs, check-only digest validation that must not rebind).
- Zero digest or error-message drift, proven by before/after payload/digest scripts per cluster (26 + 10 + full-touch coverage entries, all byte-identical) plus 31 negative-path message checks; full suite green (1865 passed).
- Known residue: a `_finite_float`/`_exact_positive`/`_exact_nonnegative` helper trio is duplicated between `episodes/retrieval.py` and `episodes/representative_retrieval.py` pending a sanctioned shared home.

## Incidental repairs

Two agent scripting incidents (CRLF line endings introduced by scripted edits) were caught via `git diff` inspection and reverted to LF before finishing; one stale test fixture that mimicked a removed inline attribute chain was updated to the canonical `durable_source_id`.
