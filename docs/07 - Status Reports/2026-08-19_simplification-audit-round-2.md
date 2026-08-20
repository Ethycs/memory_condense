# Simplification audit — round 2

**Date**: 2026-08-19 (same day as round 1, after its fixes landed in `eb387f5`/`8aa47c8`)
**Branch**: `main` at `8aa47c8`, clean tree
**Scope**: five parallel read-only sweeps — eval (including the now-landed diffuse workstream), search closure/episodes/packing, persistence/ingest/modeling/tooling, a whole-repo dead-code sweep (AST-indexed, dynamic-access-aware), and a cross-module clone sweep. Every agent read the round-1 report first; nothing listed there as done is repeated. Caller searches excluded `.agent_tmp/`, `.agent_test_tmp/`, `docs/10 - Research Log/` throughout.

## The headline

Round 1's redundancy was **twin dataclasses**; round 2's is **procedural**: transcription tables that have already drifted from their models, one algorithm cloned across layer boundaries with subtle divergences, and rituals (store probes, migrations, seals) hand-copied enough times that forgetting a step is silent. Eleven findings are correctness-adjacent, not style. Total realistic reduction: **~2,300+ lines**, of which ~550 is pure deletion.

---

## A. Correctness-adjacent divergences (fix these first)

1. **The relation ontology is declared twice and has drifted** ⚠ — `search/closure/semantics.py:21-64` (frozensets) vs `compiler.py:220-268` (tuples): same five relation families member-for-member, except `TEST_RESULT_RELATIONS` contains `"causes"` in semantics but not in the compiler (runtime-diffed). A `causes` edge is credited at walk time but never compiled into an obligation's `relation_types`. Fix: compiler imports the frozensets; decide `"causes"` deliberately. No import cycle. ~50 lines.

2. **Weighted-fair source ordering implemented twice, with two latent divergences** ⚠ — `associations/heat_diffusion.py:258-297` (`_source_fair_order`) vs `search/packing/expansion_ordering.py:287-346` (`_heat_weighted_order`): line-for-line the same deficit-round-robin, but packing clips cost to `max_expansion_tokens` while associations doesn't, and — sharper — the packing copy groups its queues by `memory_source_id or chunk.turn_id` (`:300`), **skipping `turn.source_id`**, so for turn-hydrated results its documented per-*source* fairness is actually per-*turn* fairness. The same class defines the correct `_result_source_id` twelve lines earlier. Fix: one `weighted_fair_order(...)` generic in `domain/ranking.py`; both callers keep their accessors.

3. **`source_id` unification (§C3) is ~60% done: 9 more inline copies, 3 with divergent semantics** ⚠ — six sites equal `durable_source_id` and should just use it (`application/condenser.py:324`, `application/discourse_sources.py:159`, `eval/diffuse_longmemeval.py:147`, `eval/recall_assembly.py:109`, `search/selectors/causal_choice_scorer.py:562`, `qwen_rerank.py:209`); three are divergent and need a deliberate choice (`heat_diffusion._default_source_key` :252 turn-first, `expansion_ordering.py:300` per finding 2, `span_source_queries.py:525,558` turn-only). Also delete the three one-line `_source_id → durable_source_id` shims left by round 1. Related: **three episodes/ sites detect source *conflicts* three different ways** — `episodes/retrieval.py:425-432` drops fail-open, `representative_retrieval.py:745-760` raises, `:828-835` silently continues. One `source_hints` property + one deliberate conflict policy.

4. **LongMemEval session-date parsing exists twice with three behavioral differences** ⚠ — `ingest/loader.py:229-248` vs `eval/mem0_protocol.py:117-129`: same format list over the same upstream strings, but the mem0 copy has a stricter weekday regex, no whitespace collapse, and produces **naive** datetimes where the loader produces UTC — and mem0 uses these for chronology certification. One parser (loader-side, eval imports down) with an `on_failure` knob.

5. **The codex-route temperature guard exists in exactly 1 of 6 litellm call sites** ⚠ — `eval/provider_runtime.py:112-115` guards `codex_sdk/` models against `temperature`; `eval/responder.py:69-75` passes `temperature` unconditionally (would 400 on a codex route); `judge.py`, `llm_provider.py` vary again, and content extraction is spelled four ways (judge's `:89` unguarded `.strip()` AttributeErrors on None content). One `build_completion_request(...)` + shared `_content` for the five eval sites (`application/llm_provider.py` stays independent — layering).

6. **`offline_modes.run_answer_recall` hand-writes a 124-column CSV against a 126-field model — drift verified** ⚠ — `offline_modes.py:107-488`: `evidence_source_hit` and `survives_horizon` are silently absent; `in_memory_header` emits under the header `"in_header"`. The correct pattern (`DictWriter` over `model_dump`) is at `:610-619` in the same file. ~340 lines deleted and the drift class eliminated (needs a formatter map + legacy-header alias).

7. **The benchmark and recall arms build the graph-search call as byte-identical copies** ⚠ — `benchmark.py:564-612` vs `recall_assembly.py:29-77` (34 kwargs, diff = one variable name) and `benchmark.py:713-744` vs `recall_assembly.py:189-216` (23 kwargs). Recall is documented as "the cheap predictor of the paid comparison"; a flag added to one copy silently decouples them and no test can catch it. One `graph_search_kwargs(retrieval)` / `source_search_kwargs(retrieval)` shared builder. ~120 lines.

8. **Every `db.py` migration hand-writes its own version publication (10 copies) — fail-silent hazard** ⚠ — `persistence/db.py:742-845`: each `_MIGRATIONS` entry ends with a literal `UPDATE meta SET value = 'N'…`; a forgotten tail re-applies DDL on every open. Fix: `_apply_schema_transaction` takes `version` and publishes inside the transaction it already owns. −20 lines, structurally unforgettable.

9. **The sha-digest validator family: 9 copies split on case handling** ⚠ — `domain/_discourse_identity._sha256` lowercases; `eval/cache_receipts.py:55` and `diffuse_longmemeval_runtime.py:107` casefold; the other six copies (`representative_retrieval.py:937`, five diffuse modules) reject uppercase outright — same digest string validates or fails depending on receiving class. Plus five names for one exact-int validator. One `eval/_identity.py` (re-exporting the domain validator + `exact_int`); pick the case policy once.

10. **The eval mode modules repeat one orchestration skeleton and the stress block has diverged** ⚠ — load→locked-split→stress→cap copied at `offline_modes.py:33-77`, `:494-500`, `benchmark_mode.py:10-39`, `cache_mode.py:137-169`; benchmark/cache pass `stress_question_offset` into stress composition, offline_modes defers it to `run_recall` — same flag, different stress sample per mode. Plus the reranker/selector try/finally skeleton ×3 and the provenance-hash preamble ×2. One `runtime.prepare_samples` + a context manager + `run_provenance`. ~140 lines and one divergence resolved.

11. **Association expansion guards: same preamble ×3, and `direct_tokens` uses a different denominator per arm** ⚠ — `associative_retrieval.py:39-56/233-242`, `hebbian_retrieval.py:63-80/143-149`, `heat_diffusion.py:115,334-339/585-588`: identical guard/rollback blocks, except associative computes `direct_tokens` over `bounded_anchors[:result_cap]` and hebbian over all of `bounded_anchors` — the admission budgets aren't comparable across arms. One `ExpansionGuards` + `rollback_if_over_budget`. ~90 lines.

## B. Large mechanical reductions

12. **15 eval receipt classes still hand-roll the sealing ritual** (deferred from round 1, now unblocked) — 812 lines of `__post_init__`+`identity_payload` across 7 diffuse files; four payload bodies are literally the `SealedIdentity` reflective default re-typed. Adopt the mixin (per-class overrides for the 11 projecting payloads); **digests must be proven byte-identical** (persisted receipts), and heed the plain-attribute rule documented in `domain/sealed.py` — `_diffuse_replay_contracts.py` and `_diffuse_base_derived.py:449` iterate raw `__dataclass_fields__`. ~300-400 lines.

13. **`policy_gate._policy_retrieval_identity`: 362 lines where all 95 entries are `"X": config.retrieval.X`** — plus 9 keys repeated verbatim between the qwen_rerank and qwen_feedback blocks (`policy_gate.py:166-527`). A `(gate, fields)` table + `getattr` loop; serialized identity unchanged. ~290→60 lines.

14. **`cli_config.config_from_args`: 356 lines, 94 of 94 arg-kwargs are pure identity** (`cli_config.py:9-364`) — replace the transcription with a `model_fields`-driven comprehension plus ~8 genuinely derived values; extract the two checkpoint-identity resolutions; split `cli_parser.build_parser` (151 `add_argument`s in one function) into per-group helpers. ~200 lines.

15. **`recall_measurement.measure_sample`: 523 lines, ~200 of them the fourth spelling of the coverage-selector field set** — 55 of 67 mappings are pure prefix-identity, 12 renames (`recall_measurement.py:250-450`). One `(field, key, coercer)` table; split the four independent phases. ~150 lines.

16. **`discourse_store.py` procedural triplets** — three batched chunk-scans are one algorithm ×3 (`:936-982`, `:998-1045`, `:1104-1154`; the two limit-truncation sort keys already differ) ~100 lines; the idempotent-insert guard ×4 + hydrate tail ×7 (~55); and `discourse_receipts.py` spells the 10-column snapshot row four times — once copy-pasted into `db.py:1033-1051` — with a positionally-reordered reader (~55; same hazard §D4 fixed in memory_store).

17. **The checkpoint-verification protocol exists twice in `modeling/`** — `_file_sha256` character-identical, manifest hasher differs by two lines, verify loop repeated (`qwen_prefix.py:68-226` vs `embedding.py:60-158`). One `modeling/checkpoint_identity.py`. ~70 lines. Related clone-sweep finding: **`file_sha256` has 8 copies repo-wide in two incompatible read strategies** (five 1 MiB `iter(read)` in eval + one inline in cross_encoder_selector; three 8 MiB `readinto` in modeling/selectors), including two same-named public functions in the same package (`eval.reproducibility` vs `eval.locked_split`). One canonical `file_sha256` in `domain/` (keep the `readinto` body). ~50 lines.

18. **Canonical-JSON digesting reimplemented 6× in eval, in 4 incompatible variants** — two with `allow_nan=False`, two without (NaN serializes silently), one with `default=str`, one round-trip form (`campaign_validation.py:42`, `_diffuse_replay_contracts.py:57`, `compiled_cache.py:61`, `transition_trace.py:30`, `mem0_runtime.py:173`, `_diffuse_replay_validation.py:64`) — while `domain/_discourse_identity.canonical_json/identity_sha256` is already this function and already imported nearby. ~70 lines, NaN handling becomes uniform.

19. **Round-robin/interleave: five implementations, three stall semantics** — `graph_workflow._round_robin_unique` already subsumes `query_routing.source_diverse_results:133-158`, `cross_encoder_selector.py:434-449`, `causal_choice_scorer.py:849-860` (these two also share their `source_rows` preamble verbatim), and `span_source_queries.py:315-321`. Move it down to `domain/` and express the rest as wrappers.

20. **`evidence_packet.pack_evidence_plan`** — the 7-element beam objective spelled twice (`:162-178` vs `:196-206`, one hand-negated; the `reversed(bundle_ids)` element is dead — dedup keys guarantee equality) and the render→count→check triad ×4 with a **doubled render per rejected bundle** in an O(candidates) loop (`:613-621`). One `_state_order_key` + one `_measure`. ~45 lines plus real work saved.

21. **Closure engine ritual** — the store-probe/validate/witness block ×7 (`engine.py:375-1021`; overflow message ×4 incl. `scope_scan.py:62`): one `_probe(...)` helper makes a forgotten witness structurally impossible (today a missing witness silently degrades `completion()`'s claims via `results.py:242-245`). ~90-110 lines. Also: the episode-chunk-set comprehension ×3, the throwaway-`_UnitRoute`-to-read-`.connected` idiom ×2, `_Walk.capped` write-only, 3 unused imports.

22. **Ingest/tooling/modeling mechanical set** — loader session-flattening skeleton ×2 (~50, plus the boundary-turn template producer/parser split with `transcript_store._SOURCE_METADATA_RE`); tooling Qwen-bootstrap ×4 with inconsistent flag names (~60, keep `--prefix-layers` alias); `qwen_prefix` twin dtype tables / twin layer validators / recomputed shard set / per-call `_require_torch_stack` in `output_for_head` (~50); five positional `CorpusSource(...)` constructions + the still-open §D3 shared source-location fields (~30); `validator.py` five copy-pasted accept/reject blocks (~25); `transcript_store` turn-SELECT ×4; `embedding.embed_chunks` field-by-field rebuild → `model_copy` (11→1); `partition_workflow` ordering key inlined twice next to its own named function (~15); SQL constants `COALESCE(t.source_id, t.turn_id)` (20+ sites) and the indexed-chunk predicate (~10 sites) → two constants in `db.py`; `_NUMBER_WORDS` defined identically in three search modules (~40); `_cosine` copied across `transition_policy`/`transition_replay` (the import path already exists); `_canonical_callable_code`/`_canonical_code_object` byte-identical pair (~18); `min_max_normalize` and softmax local variants → parameterized domain helpers; `eval/runner._prompt_tokens` ≡ `benchmark._message_content_tokens`; §E1 residual: the co-access `neighbors` fetch-and-validate preamble still parallel (~45).

23. **Remaining deferred wrappers (now unblocked)** — `RepresentativePolicyFactory` **name collision, both in `__all__`** (`diffuse_longmemeval_analysis.py:120` Callable alias vs `diffuse_longmemeval_runtime.py:830` class): rename the class; `ExactLegacyDiffuseInputs.__post_init__` re-verifies digests its sole constructor just computed (`diffuse_longmemeval_inputs.py:424-465`); `_question_probe` still a character-level copy (`_analysis.py:902` vs `_inputs.py:313`); `runtime_controls._load_coverage_selector`'s four branches share a guard/tail shape a backend registry removes (~100 of its 313 lines). ~170 lines total.

## C. Dead code (whole-repo sweep; ~550 lines of pure deletion)

All verified with word-boundary greps plus dynamic-access checks (`_EXPORTS` lazy table, `getattr` string literals, monkeypatch targets, pixi tasks, MCP registrations):

- `search_heat_associative` + `expand_heat_associative` (`retrieval_workflow.py:206-311`, 106 lines, zero callers; engine `expand_heat_diffusion_results` stays live) — needs one architecture-doc line edit.
- `search_hebbian` (`:313-372`, 60 lines, zero callers; two doc prose lines) — cascades: `expand_hebbian` and `observe_retrieval_access` become test-only (63 more lines, author call).
- `QwenLiveHeadMemory.retrieve_candidates` (53 lines, zero refs; cascades `HeadKVStore.indices_for_episode_ids` to test-only).
- `QwenMemoryLinker.link_into_graph` (22 lines + dead import; not in the `_OWNED_CRITICAL_NAMES` allowlist; cascades `prune_neighbors` to test-only).
- `judge_response` (test-only wrapper; production uses `_with_usage`) + its tests.
- Never-passed parameters: `episodes_for_source(start_sequence=, end_sequence=)` (~20 lines incl. Protocol stubs and test-fake filter logic), `expand_context_associations(max_chunk_token_increase=)` (~11), `link(include_transport_signature=)` (~8), `compile_query_program(manual_program=)` (~5), `compile_cav_signatures(conceptual_spans=)` (~7).
- `HybridQueryMixin._load_chunk`/`_load_turn` (+4-name import cascade), `DiscourseStore.episodes_for_chunks` (13), `incident_units` (test-only, not in the closure Protocol), `benchmark._message_content_tokens`, `Mem0IngestResult.pairs_added`, `CoaccessUpdate.concepts_observed`, `SourceChunkStream.all_chunk_ids`, `score_episode_surprises` alias, four dead surprise-facade aliases, six single-use unused imports, `run_diffuse_treatment_sample` (92-line public function, zero callers — wire up or delete), `_message_prompt_token_proxy` pass-through.

Checked and kept (would false-positive in a naive sweep): `pin_memory`/`memory_stats` (live via `@mcp.tool()`), experiment-rig entry points, `retained_token_state_bytes` compat keys (read by experiment_rig), `campaign.transcript_tokens` and `recall.print_recall_report` facade seams (monkeypatched/imported through), `tokenizer_proxy_identity` (reflective read), the `_EXPORTS` table (architecture-test-asserted public API).

## D. Structural (little line change, large comprehensibility gain)

- `expansion_assembly._build_expansions` — 707 lines, five separable phases, `trace_by_id` mutated in six places, the diagnostic-row literal written twice with a silent field difference. Extract phases around one `_ExpansionPass` state object + one `_trace_row` factory; densely covered by `test_context_packer.py`.
- `closure/engine._walk` — 473 lines; move into a `_ClosureWalk` owner with `seed/scope/expand/receipts`. `engine.py` is 48 lines under the 1300-line architecture-test tripwire.
- `campaign_merge.merge_benchmark_reports` — 673 lines, nine accumulators mutated three levels deep, currently untestable except end-to-end; decompose into per-stage validators + a `_MergeAccumulator`.
- `EpisodeRepresentativeRetrievalPlan` built twice with 15 identical kwargs (`representative_retrieval.py:646-666` vs `:705-728`); two divergent Qwen-linker identity extractors (`:946-994` vs `qwen_episode_signal.py:396-450`) → one `qwen_linker_identity(strict=)`; the authoritative span ordering hand-spelled three times → route through `evidence_span_sort_key`.
- `INICoverageSelector` should accept-and-ignore the three `active_partition_*` kwargs so it satisfies its own Protocol — then the load-bearing `inspect.signature` negotiation in `expansion_assembly.py:125-155` (~28 lines) and its three partial-capability test doubles can go. Author decision.

## Explicitly checked and not recommended

`db.py._MIGRATIONS[2]`'s apparent schema copy (deliberate pre-v4 column set, documented and tested); `_compile_obligations`'s 125-line if-chain (a data table is a taste call); `SweepArm.__post_init__` (distinct guards, not duplication); experiment-rig's 25-key artifact dict (frozen schema); retry/backoff and token counting (already centralized); SQL placeholder building (idiomatic); the round-1 skips that remain author decisions (§G2 QuestionRecall nesting, §C1 signature restructuring, campaign_validation pydantic merge).

## Suggested order of attack

| Phase | Items | Character |
| --- | --- | --- |
| 1 | A1-A11 | Correctness-adjacent; mostly small, each closes a live divergence |
| 2 | C (dead code) | Pure deletion, immediately shrinks everything after it |
| 3 | B12 (receipts) → B13-B15 (transcription tables) → B6 CSV | The big mechanical wins; receipts need digest-stability proof |
| 4 | B16-B23 | Package-local mechanical set |
| 5 | D | Structural decompositions, best done on a quiet tree |
