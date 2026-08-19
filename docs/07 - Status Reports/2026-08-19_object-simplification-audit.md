# Object-simplification audit — full-project sweep

> **Update (same day): implemented.** See "Implementation status" at the end of this document. All fixes applied against `4974dbd` in the working tree (uncommitted); full suite green at 1841 passed / 1 skipped (baseline was 1840 passed / 1 skipped; the count shifted by new parallel-workstream tests and one test deleted with its class).

**Date**: 2026-08-19
**Branch**: `main` at `ecbb9dd`, working tree untouched
**Scope**: all of `src/memory_condense/` (~66k lines, 177 files), audited by four parallel read-only reviews (eval / associations / search / application+domain+persistence+ingest) plus a direct read of `graph_workflow.py` and `ingest_workflow.py`. Every finding below was verified against the actual class definitions and at least one usage site. No code was changed.

## The headline

The project's redundancy is **not** serialization plumbing — it is the same three habits repeated across packages:

1. **One protocol, hand-copied everywhere**: the "self-sealing receipt" ritual (`identity_payload()` + verify-or-set `*_sha256` in `__post_init__`) is written out in **23 dataclasses across three packages**. One shared mixin removes ~600 lines.
2. **One field set, spelled in 3–5 places**: several report/snapshot shapes exist simultaneously as a dict literal, a dataclass, and a field-by-field transcription between the two.
3. **One algorithm, two record types**: whole subsystems (co-access graphs, CAV neighbor scoring, head edges, selection reports) are maintained twice behind near-twin classes.

Two of the findings are latent-bug-adjacent, not just style (§C3 source-id fallback divergence, §D4 memory-store column-order fragility) — worth fixing first.

---

## A. Cross-package: the sealed-receipt ritual (~600 lines, highest leverage)

Every one of these classes ends `__post_init__` with the identical 5-line block (`expected = identity_sha256(self.identity_payload(...)); if self.X_sha256 and != expected: raise; object.__setattr__(...)`) and hand-writes an `identity_payload()` that re-lists field names `__dataclass_fields__` already knows:

| Package | Classes | Evidence |
| --- | --- | --- |
| `domain/discourse.py` | 7 (`Episode` :276, `DiscourseSnapshot` :466, `ArtifactCoverageReceipt` :500, `QueryProgram` :600, `ClosureScopeWitness` :805, `ClosurePlan` :1002, `ClosureReceipt` :1238) | 84 `object.__setattr__` calls, 82 `_nonempty`/`_sha256` calls — one per field per class |
| `eval/` (diffuse\* modules) | 15 receipt classes across 7 files (`diffuse_compilation.py:103,167`, `diffuse_longmemeval.py:189`, `diffuse_longmemeval_analysis.py:361,511,640,674`, `diffuse_longmemeval_inputs.py:366`, `diffuse_longmemeval_matched.py:66,132`, `diffuse_longmemeval_runtime.py:407,1106`, `diffuse_longmemeval_runtime_matched.py:122`) | 15 hits for `identity_payload(include_receipt=False)`; four payload bodies are literally `{name: getattr(self, name) for name in self.__dataclass_fields__ if name != "receipt_sha256"}` |
| `search/episodes/surprise_models.py:34-75` | `AttentionHeadSurpriseReceipt` | `identity_payload` (:204-248) re-lists all 39 fields by hand |

**Fix**: one `SealedIdentity` mixin (shared module, e.g. `domain/sealed.py`) providing `_seal(field)` and a default `identity_payload()` driven by `__dataclass_fields__` minus a class-level `_DERIVED` frozenset, with a per-field normalizer map replacing the 84 `object.__setattr__(self, "x", _nonempty(self.x, "x"))` lines. The reflective approach is already proven **inside this codebase**: `ClosurePolicy.policy_sha256` (`domain/discourse.py:670-677`) and `EpisodeRepresentativeRetrievalPolicy.policy_sha256` (`search/episodes/representative_retrieval.py:245-249`) both do it. Also fold in the copy-pasted validators: `_digest` ×5, `_positive_int` ×3 (two spellings) across the diffuse modules; `LongMemEvalDiffuseQueryReceipt.__post_init__` (`diffuse_longmemeval.py:250-261`) inlines the digest check twice instead of calling any of them.

## B. Cross-package: constant-valued "fields" that are really invariants

`retained_request_token_state_bytes` / `retained_transformer_state_bytes` and friends are **settable fields whose `__post_init__` raises unless they equal 0**:

- `ingest/discourse_linker.py:160-164`; `domain/discourse.py:1088,1116-1117` (then written back as literal `0` at :1218); `persistence/discourse_store.py:1289-1290`
- Six selector report classes: `search/selectors/coverage_models.py:70`, `cross_encoder_selector.py:128,147`, `causal_choice_scorer.py:184,208`, `qwen_rerank.py:54`; plus `episodes/representative_retrieval.py:359` and `surprise_models.py:197`
- The only explicit assignment anywhere in `src/` is `retained_transformer_state_bytes=0` (`cross_encoder_selector.py:739`)
- Same pattern, different mechanism: `AttentionHeadSurpriseReceipt.format/algorithm/score_formula/head_similarity_algorithm` default to module constants and raise if they differ (`surprise_models.py:147-154`)

**Fix**: these express a *type-level* invariant ("this pass retains no transformer state"), not per-instance data. Replace with class-level constants / `ClassVar` included in the identity hash, and keep the one runtime check that validates an *external* dict (`cross_encoder_selector.py:521-524`). Removes ~10 fields from ~10 dataclasses plus their constructor threading. (`EpisodePublication.returned_signal_transformer_state_bytes` at `discourse_workflow.py:230` is genuinely tri-valued and should stay.)

## C. Application layer

### C1. `search_hybrid_graph` — 38 keyword parameters, 60 lines of validation
`application/graph_workflow.py:28-132`. The parameters cluster into obvious groups (facet, role weights, TF-ISF/HSC activation, partition routing, attention feedback), each of which could be a small frozen config dataclass validating itself in `__post_init__`. The role-weight triple alone is threaded into **6** identical `role_aware_results(...)` calls; the "extract source_id → setdefault anchor → max score" accumulation loop appears **4** times (:405-427, :412-419, :589-601, :782-792); the round-robin interleave loop appears twice (:359-375, :754-771). Six `last_*_report` side-channel dicts are mutated throughout — one telemetry object would do.

### C2. The active-partition field set is spelled **four** times
- `partition_workflow.py:175-207` builds the 28-key report dict (updated at :498-526, :629-637)
- `graph_workflow.py:1007-1093` — **86 lines** copying that dict into `ActivePartitionRoutingSnapshot`, every kwarg name byte-identical to the dict key
- `condenser_contracts.py:59-90` declares the same 19 names as dataclass fields; `pack_fields()` (:115-144) re-lists them **again** as a string tuple

The snapshot has exactly one construction site (`graph_workflow.py:1008`) and two consumers (`condenser.py:264,371,391`). **Fix**: `ActivePartitionRoutingSnapshot.from_report(...)` driven by `__dataclass_fields__` with annotation-based coercion; `pack_fields()` becomes a derived `getattr` comprehension. Bonus: `condenser.py:389-404` currently inspects `self._packer.pack`'s signature at runtime to decide what to forward — a typed `active_partition` parameter removes the reflection.

### C3. `source_id` derivation: four copies, **two divergent behaviours** ⚠
`result.turn.source_id or result.turn.turn_id` is inlined at ~14 sites (`retrieval_workflow.py:497,572,615`, `graph_workflow.py:244,403,407,415,423,593,694,785`, `query_routing.py:219`, `packing/expansion_ordering.py:282-288`, `packing/derived_scalar.py:333-338`, `selectors/evidence_features.py:15-20`, `cross_encoder_selector.py:154-159`) while `query_routing._retrieval_source_id` (:129-135) already exists. The copies **disagree on the fallback**: the `or`-chain falls through to `chunk.turn_id` when `source_id` is falsy; the if-form returns `turn.turn_id`; `_retrieval_source_id` consults `memory_source_id`. Same intent, different answer for turns without a `source_id`. **Fix**: one public `source_id` property on `RetrievalResult` (or a helper in `packing/source_provenance.py`), one deliberately chosen behaviour, all 14+ sites routed through it. `ContextPacker._bind_source_metadata` (`context_packer.py:213-228`) already takes it as an injected callable — the seam exists.

### C4. `last_source_companion_report` — the same ~18-key dict literal in three places
`condenser.py:218-237`, `source_companions.py:247-271` (`empty_report`), `source_companions.py:744-785`. All three must agree on names and defaults; drift is silent (consumers use `getattr(mc, ..., {})`). **Fix**: one frozen `SourceCompanionReport` dataclass with defaults and `as_dict()`. Relatedly, `CrossEncoderCompanionReport` and `CausalChoiceCompanionReport` (`cross_encoder_selector.py:135-151`, `causal_choice_scorer.py:191-212`) share 7 of 10 fields and land in the same slot.

### C5. Delegation and dead aliases
- `discourse_workflow.py:82-215` `_ArtifactScopedDiscourseStore`: 17 hand-written forwarders, only 5 add behaviour; untyped signatures silently drift from `DiscourseStore`. Use `__getattr__` delegation + explicit overrides, or wrap the existing `search/closure/store.py:67` Protocol. ~90 lines.
- `persistence/discourse_store.py:171-223`: of six alias methods, **five have zero production callers** (`register_artifact`, `publish_batch`, `put_episodes`, `put_units`, `put_relations`); `put_artifact` is test-only. Delete; migrate ~8 test call sites to `publish`. `relations_incident_to` (:1133) is a one-line unwrap used by one test.
- `retrieval_workflow.py` + `graph_workflow.py`: four mechanical `search_X`/`expand_X` pairs (~300 lines) where each `search_X` is guard → `search_hybrid` → `expand_X` with 10-15 kwargs retyped, and the `SAFE_ASSOCIATION_*` defaults retyped in **six** signatures. One `_search_then_expand` helper + a shared `AssociationGuards` frozen dataclass.

### C6. `ingest_many` tuple-length dispatch
`application/ingest_workflow.py:78-119` accepts a union of 3/4/5-tuples and length-dispatches to normalize. A tiny `TurnRecord` dataclass (or accepting keywords) deletes the whole dispatch.

## D. Domain & persistence

1. **`RetrievalResult` / `MemoryResult` duplicate the diagnostic field group** — `route`, `consolidation_score/anchor/support` with identical constraints (`domain/schemas.py:307,328-330` vs :350-353). Shared mixin or one nested `consolidation` sub-model. Also `CreateOp` (:199-208) is a strict subset of `MemoryItem` (:151-177) mapped across by hand in `MemoryStore.create` — a `MemoryItem.from_create(op)` puts the mapping next to the fields.
2. **`persistence/discourse_evidence.py:49-79`** — `_SourceRow` re-declares the chunk⋈turn join as an 11-field dataclass duplicating `Chunk`+`Turn`, with its own hand-built identity hash; the 10-kwarg `EvidenceSpan(...)` construction is then written out twice more (:242-253, :276-287).
3. **`ingest/corpus.py:109-126`** — `CorpusRecallEpisode` / `CorpusRecallQuestion` share three source-location fields; one base or a shared `SourceLocation` removes the divergence risk in `build_conversation_recall_slice`.
4. **`persistence/memory_store.py` — three parallel positional field lists for one row** ⚠: `_ITEM_COLUMNS` (:48-52), the INSERT (:594-620, *different column order*), and `_row_to_item` (:709-736, positional `row[0..13]` with two indices swapped relative to the SELECT). This works today but is exactly the fragility positional row-mapping invites. **Fix**: one `_COLUMNS` tuple deriving SELECT list, INSERT placeholders, and a `dict(zip(...))`-based reader with a small coercion map. Same treatment for `Provenance` round-tripping (:622-632, :678-707).

## E. Associations — duplicated algorithms behind twin record types

1. **`LiveConsolidationStore` is a near line-for-line clone of `HebbianAssociationStoreMixin` (~600 duplicated lines)**. `consolidation.py:299-936` vs `hebbian_store.py:18-532`: `_decayed_mass`, edge scoring (same cosine × freshness, same comment), observe/reinforce (identical validation, SHA-256 receipt, `combinations(selected,2)`, upsert, receipt trim), neighbors (identical noisy-OR accumulation), prune, stats. The value objects twin too: `ConsolidationUpdate` ≡ `HebbianUpdate` (`association_models.py:150-158`), `ConsolidationNeighbor` ≡ `StoredHebbianNeighbor` (:161-170) plus one field; `context_activations` ≡ `retrieval_concept_activations` (same `1.0/sqrt(rank)`); `_canonical_json` redefined identically. Both are live (`condenser.py:133,303`; `retrieval_workflow.py:392,426`) — genuine parallel maintenance. **Fix**: one generic `CoaccessGraphStore` parameterized by a table/column descriptor; one `CoaccessUpdate` / `CoaccessNeighbor` (`causal_count` defaulting 0).
2. **CAV-neighbor scoring implemented twice; one copy has no production caller**. `cav_memory.py:63-127` (`CAVLinkIndex`, in-memory) vs `association_artifacts.py:196-291` (SQLite): identical algorithm down to the `0.1 * shared/union` bonus and the sort key. `CAVLinkIndex`/`CAVNeighbor` are referenced only by the `head_memory.py` facade and one test. Delete (retarget the test at `AssociationStore.cav_neighbors`).
3. **`StoredHeadEdge` vs `HeadAssociationEdge`** (`association_models.py:121-147` vs `head_memory_models.py:122-130`): same directed QK/OV edge, fields renamed; the evidence-weighted merge is implemented twice (numpy `association_edges.py:97-117`, torch `head_association_graph.py:76-84`, same formula) and the prune ranking at `head_association_graph.py:198-217` re-inlines `StoredHeadEdge.utility`. Collapse to one `HeadEdge` + one `merge_evidence`.
4. **`NestedMemoryInspection` is `MemoryLinkResult` minus one field with two fields renamed** (`head_memory_models.py:94-113`); consumers already treat them interchangeably. Delete it; return `MemoryLinkResult` with `source_cav_signature=()`.
5. **Four copy-pasted result-building blocks** in `qwen_live_memory.py` (:297-321, :371-394, :432-453, :493-507), including the fifth copy of the `tuple(float(v) for v in x.tolist())` ternary in the package. One private `_result(...)` method.
6. **`AssociativeMemoryCandidate.metadata` is an untyped dict used as a struct**, built with the identical 10-key literal twice (`heat_diffusion.py:386-409`, :459-484), indexed downstream with string literals and casts, and built with a *different* shape by `associative_retrieval.py:152-157`. Promote the provenance keys to typed optional fields.
7. **Write-only / throwaway fields**: `HeadAddress.head_scores`/`concept_scores` are assigned (`head_kv_store.py:222-233`) and never read anywhere in src or tests — drop them. `StoredCAVSignature`'s `created_turn`/`access_count`/`last_access_turn` are decoded, cast, and never read; the ingest path (`ingest_workflow.py:195-198`) only needs an existence check — add `has_signature() -> bool` (`SELECT 1`).
8. **Facade imports drag the Qwen stack into model-free paths**: `head_memory.py` imports the Qwen modules at module scope; `associative_retrieval.py` and `heat_diffusion.py` import through it for two symbols that live in `head_memory_models`/`associative_composition`; `consolidation.py:213` already works around this with a comment-justified lazy import. Point the three sites at the real modules. `association_store.py` is a second pure re-export facade over `association_repository` — consider merging the module names.
9. **`StoredCAVNeighbor.source_id` is a shape union**: always `None` from `cav_neighbors`, always set by `concept_members`, which then does an unconditional `str(hit.source_id)` that would produce `"None"` if crossed. Split the types or make the field required; extract the ~35 shared lines of artifact-scan scaffolding.

## F. Search & packing

1. **`CrossEncoderSelectionReport` is a 52-field clone of `CoverageSelectionReport`** (`cross_encoder_selector.py:65-132` vs `coverage_models.py:44-135`), populated by a **175-line field-by-field transcription** of the grouper report's own `model_dump()` (:567-741) — which is then *also* stored whole at :738. Consumers can't tell the classes apart (`coverage_closure.py:31-36` reads via a Mapping-or-object shim). One union `SelectionReport` (or a shared `_FrontierScopeFields` base for the 30 shared `frontier_*`/`active_partition_*` fields) + `dataclasses.replace(...)` deletes ~170 lines.
2. **Five construction sites repeat the same 15-field "zeros" preamble** for `CoverageSelectionReport` (`ini_coverage_selector.py:283-302,316-344,513`, `prefix_selector.py:355-409`, `prefix_reservation.py:859-912`) — and they have **already drifted**: `_bypass` hardcodes `requires_completeness=False` while `_fallback` passes `program.requires_completeness`. Add a `CoverageSelectionReport.uninspected(program, ...)` classmethod taking the `SetProgram` whole.
3. **`SetProgram.operator` and `.requires_completeness` are pure functions of `quantifier`+`ordering`** (`set_program.py:272-285`, :358-361), with `cardinality ⟺ FIXED` a third hand-maintained invariant. All 12 non-facade call sites branch on `quantifier`, and every `operator` read is `.value`-as-string. Delete the `SetOperator` enum; make both properties.
4. **Seven identical `def model_dump(self): return asdict(self)` bodies** across the selector reports — one mixin, or drop the method and call `asdict` at the 3 call sites.
5. **Timestamp parsing ×3 with the same regex literal** (`evidence_features.py:29-54`, `source_provenance.py:13-41`, `derived_scalar.py:109-113`); consolidation is half-done (`coverage_closure.py:9-11` already imports the provenance one). Keep `provenance_timestamp_key` with an `allow_bare_year` flag.
6. **Small objects with no second consumer**: `_PostCoverageClosure` built once and splatted into a dict two lines later (`coverage_closure.py:14-20`, `expansion_assembly.py:563-624`); `_RawAssignment` → `CandidateAssignment` pydantic-to-dataclass translation with a single bridge and single consumer (`coverage_models.py:12-41`, `ini_coverage_selector.py:156-177`) — collapse to one model with computed properties.
7. Runners-up: `_PreparedCoverage`/`_ScoredCoverage` untyped 17/22-field parameter bags signalling failure by returning a different type (`prefix_models.py:49-96`, `prefix_pipeline.py:50-54`); `SimilarityRetriever` empty class over three mixins kept alive by 20 lines of deliberately-unused monkeypatch-seam imports (`indexes/retrieval.py:6-49`); `BoundaryRefinement` = `BoundaryProposal` + 2 fields, constructed field-for-field when no refiner is set (`episodes/boundaries.py`, `builder.py:160-168`); `ArtifactUnitScan` with three near-identical construction sites (`closure/scope_scan.py:87-127`); `_exact_int` defined twice identically (`episodes/retrieval.py:519`, `representative_retrieval.py:936`).

## G. Eval

1. **Mirrored "compatibility alias" fields**: every prompt-token metric stored twice from the same local (`benchmark.py:167-169,202-205,233-250`; `mem0_models.py:165-168,233-235,251-267`; plus `SourceRef.pair` ≡ `batch_index`, `pairs_added` ≡ `batches_added`). Keep one stored field; emit the legacy spelling only in serialization (`@computed_field`/serializer alias). Kills the update-one-forget-the-other bug class.
2. **`QuestionRecall` has 140 flat fields, ~80 of them one optional sub-object**: 62 `coverage_selector_*` fields populated only when a selector ran; `RecallReport.model_post_init` (`recall_models.py:234-358`) spends 125 lines re-aggregating them behind a 7-clause "was this active" disjunction. The right shape already exists in the same file (`AnswerValueCoverage`, whose six fields are *also* flattened in). Introduce `CoverageSelectorTrace | None` and use `AnswerValueCoverage` directly; "is this metric meaningful" becomes `is not None`.
3. **Compatibility facades that re-export ~90% never-imported private names**: `campaign.py` (~40 re-exports, 35 of them underscore-private with zero importers), `mem0_adapter.py` (~70), `recall.py`, `__main__.py` (~30 aliases) — and the pattern has grown a `sys.modules.get(...)`/`getattr` reflection hack to honour facade monkeypatches (`mem0_runtime.py:35-39`, `mem0_longmemeval.py:69`).
   **Important scoping**: the facades themselves are a *deliberate, test-enforced* pattern — `tests/test_architecture.py:65-84` defines `_WORKFLOW_FACADES` with per-file line budgets and `test_decomposed_facades_remain_small` asserts each stays under them ("Compatibility modules orchestrate; they do not regrow implementations"). So the fix is **trim the re-export list, not delete the module**. `mem0_adapter.py` is genuinely load-bearing (7 external importers across `tools/mem0_eval/` and tests); `campaign.py`'s private re-exports have zero importers. The line-savings estimate holds only for the `_`-prefixed re-exports.
   **Zero-risk first step**: four modules inside the package import their own siblings *through the shim* instead of directly — `consolidation_replay.py:30`, `sufficiency.py:25`, `transition_trace.py:22`, `__main__.py:40` all pull `contains_answer`/`best_f1`/`_assemble`/`run_recall`/`print_recall_report` from `eval.recall`, when those symbols live in `answer_value_coverage.py`, `recall_assembly.py`, `recall_measurement.py`, and `recall_reporting.py`. Repointing them makes the dependency graph honest and lets the facade shrink to what external callers actually need. (The frozen snapshots under `docs/10 - Research Log/data/` and `.agent_tmp/` should not constrain the trim.)
4. **Twin classes / throwaway wrappers**: `BinDelta` ≡ `ConversationDelta` modulo the key field, with the derived-field math duplicated at both construction sites (`analysis.py:39-62,171-198`) → one `PairedDelta.from_pair`; `RepresentativePolicyFactory` holds one field and re-spells 11 config fields the config already owns (`diffuse_longmemeval_runtime.py:811-843`) — and its name collides with an unrelated `Callable` alias in `diffuse_longmemeval_analysis.py:120-122`; `ExactLegacyDiffuseInputs` re-verifies in `__post_init__` the digests its sole constructor computed 20 lines earlier (`diffuse_longmemeval_inputs.py:424-506`); `_LazyLocalINICoverageSelector` is an empty subclass used as a docstring (`runtime_controls.py:256-257`); `_question_probe` duplicates the gold-blind projection character-for-character (`diffuse_longmemeval_analysis.py:897-902` vs `diffuse_longmemeval_inputs.py:313-317`).
5. **Deliberately not recommended for mechanical merge**: `campaign_validation.py:56-111` hand-rolls JSON validators that look replaceable by `BenchmarkRunResult.model_validate`, but the module is documented as fail-closed independent re-validation of untrusted artifacts and is *stricter* than pydantic (rejects `bool`-as-`int`, `allow_nan=False`). Author decision required.

---

## Suggested order of attack

| Priority | Item | Why first | Est. lines |
| --- | --- | --- | --- |
| 1 | §C3 unify `source_id` derivation | Latent behavioural divergence, tiny fix | ~30 |
| 2 | §D4 memory-store column lists | Fragile positional mapping around live data | ~60 |
| 3 | §F2 `CoverageSelectionReport.uninspected` | Already-drifted defaults (`requires_completeness`) | ~90 |
| 4 | §A `SealedIdentity` mixin | Biggest single win; enables §C2 and §F1 patterns | ~600 |
| 5 | §C2 active-partition `from_report` | 86-line copy block, one construction site | ~120 |
| 6 | §E1 unified co-access store | Largest algorithm duplication | ~600 |
| 7 | §F1 selection-report union | 175-line transcription | ~170 |
| 8 | Dead code (§C5 aliases, §E2 `CAVLinkIndex`, §E7 write-only fields, §G4 wrappers) | Pure deletion, test-only migrations | ~250 |
| 9 | §C1 parameter objects for `search_hybrid_graph` | High conceptual win, wide blast radius — do after the above stabilize | n/a |

Total realistic reduction: **~2,000+ lines** with no behaviour change, plus the removal of three silent-drift hazards.

## Blast-radius notes (from a follow-up sweep outside `src/`)

- **§E2 is not pure deletion**: `CAVLinkIndex` is described in prose at `docs/02 - Implementation/03 - Qwen3 Prefix Attention Lab.md:151-152` ("only float32 concept coordinates and Concept↔Episode membership") — that paragraph needs updating when the class goes.
- **§E3 has an out-of-tree caller**: `docs/10 - Research Log/data/2026-08-16-build-session-baseline/cc_notes_live_benchmark.py:27,307` imports and constructs `HeadAssociationGraph`. It is a frozen research-log artifact, so leaving it broken-by-design is probably right — but make that choice deliberately when collapsing to one `HeadEdge`.
- **Exclude `.agent_tmp/` and `.agent_test_tmp/` from impact greps**: `.agent_tmp/frozen_v3/` is a pre-split snapshot of the associations modules (`association_store.py`, `head_memory.py`) containing the very classes being removed. Repo-wide greps will report them as live callers; they are not.
- **§E1's rename needs a facade alias**: `application/retrieval_workflow.py:13` imports `HebbianUpdate` from `association_store` (the facade, not `association_models`) and uses it in a return annotation at :420. A `CoaccessUpdate` rename needs a back-compat alias in `association_store.py`'s `__all__`.

---

## Implementation status (applied 2026-08-19, uncommitted)

Applied by five package-scoped agents on disjoint file sets plus an application-layer pass, against `main` at `4974dbd`. Baseline before edits: 1840 passed / 1 skipped / 13 deselected. Final: **1841 passed / 1 skipped / 13 deselected, zero failures** (the count shifted by the parallel diffuse-replay workstream's new tests, minus tests migrated or deleted with their classes). The in-flight diffuse eval workstream (`diffuse_*`, `_diffuse_base_*`, `_diffuse_replay_*`) was deliberately left untouched, so §A's 15 eval receipt classes and §G4's diffuse wrappers are **deferred** until that work lands. Net delta ≈ **−830 lines** in existing files, +~340 in two new shared modules (`domain/sealed.py`, `associations/coaccess_graph.py`).

**Done**
- §A (domain + search parts): `SealedIdentity` mixin adopted by discourse.py's 7 sealed classes and `AttentionHeadSurpriseReceipt`; payload/digest equality verified byte-for-byte by scratch harnesses against pre-refactor instances. Normalizer boilerplate collapsed via `normalize_fields`. discourse.py net −157.
- §B: left in place by design — removing the constant fields would change dump shapes. Only the redundant explicit `=0` threading was dropped where safe.
- §C1 (internal only): role-weight closure, `_accumulate_source_activation`, `_round_robin_unique` (a `stop_on_stall` flag preserves the two loops' different stall semantics). The 38-kwarg public signature was deliberately kept — restructuring it would break/invalidate existing callers and locked eval configs; that remains a follow-up decision.
- §C2: `ActivePartitionRoutingSnapshot.from_report()` + derived `_SCAN_FIELDS`; the 86-line transcription and the retyped `pack_fields` tuple are gone.
- §C3: `RetrievalResult.source_key` (turn-first) and `RetrievalResult.durable_source_id` (memory-source-first) now name the two behaviors explicitly; all 14+ sites route through one of them.
- §C4: `default_source_companion_report()` is the single authoritative shape (rejects unknown keys); all three producers use it.
- §C5: the five dead `DiscourseStore` aliases + `relations_incident_to` deleted (tests migrated to `publish`); `_ArtifactScopedDiscourseStore`'s nine mechanical scope-injectors collapsed to two method factories while keeping the fail-closed allowlist explicit (no `__getattr__` — that would fail open). The `search_X`/`expand_X` kwarg-forwarding was **skipped as cosmetic**: the lists are the typed public API surface.
- §C6: tuple-length dispatch replaced by padding + one validation.
- §D1/D4/D2: `ConsolidationDiagnostics` base for RetrievalResult/MemoryResult; `MemoryItem.from_create` adopted by `MemoryStore.create`; memory-store rows single-sourced from `_COLUMNS` (the INSERT/SELECT order mismatch is gone); `_SourceRow.span()` dedupes the EvidenceSpan constructions.
- §E: shared `coaccess_graph.py` engine behind both the consolidation and Hebbian stores (receipts byte-identical; `CoaccessUpdate` with `HebbianUpdate`/`ConsolidationUpdate` aliases); `CAVLinkIndex`/`CAVNeighbor` deleted (doc paragraph updated); shared `evidence_weighted_mean`/`qk_transport_utility`; the four `qwen_live_memory` result blocks and the duplicated heat-diffusion metadata literal collapsed; write-only `HeadAddress` fields dropped; `has_signature()` added and adopted by ingest; facade imports repointed so model-free paths no longer load Qwen; the `"None"`-key hazard in `concept_members` guarded. **§E4 (NestedMemoryInspection) skipped**: `isinstance` checks and field-name differences make a bare alias unsafe; needs a coordinated cross-package change.
- §F: the 175-line report transcription replaced by grouped mirrored-field tuples; `CoverageSelectionReport.uninspected()` (the `_bypass` drift is preserved and commented, not silently changed); `SetProgram.operator`/`requires_completeness` are now properties (`SetOperator` kept — enum identity is used); `ReportDumpMixin` replaces seven identical `model_dump` bodies; one timestamp parser (with `allow_bare_year`/`assume_utc` capturing the real divergences, equivalence proven over 17 edge cases); `_PostCoverageClosure` and `_RawAssignment` collapsed.
- §G1: alias metrics now single-stored (pydantic `@computed_field` / dataclass `init=False` derived fields); serialized keys and values verified identical (JSON key *order* shifts on three benchmark models; all digest paths sort keys).
- §G3: four intra-package shim imports repointed to the real modules; ~30 private re-exports trimmed from `campaign.py` and ~32 from `mem0_adapter.py` (everything the 7 real external importers use is kept); the `mem0` `sys.modules` reflection hack removed after verifying nothing monkeypatches it (campaign's analogous hack IS load-bearing and stays).
- §G4 (non-diffuse): `_paired_delta_fields` dedupes the BinDelta/ConversationDelta math (no field renames — dump shape kept); `_LazyLocalINICoverageSelector` deleted.

**One incident worth remembering**: annotated `ClassVar`s inside a dataclass body land in the raw `__dataclass_fields__` mapping as pseudo-fields. The parallel replay workstream rebuilds snapshot identity payloads by iterating that raw mapping, so the mixin's per-class `_SEAL_FIELD`/`_SEAL_MISMATCH` annotations broke its digest check. Fixed by declaring the seal knobs as plain un-annotated class attributes (they now vanish from `__dataclass_fields__` entirely); the rule is documented in `domain/sealed.py`.

**Deferred** (blocked on the in-flight diffuse workstream): §A's 15 eval receipt classes and shared `_digest`/`_positive_int` validators; §G4's `RepresentativePolicyFactory` and `ExactLegacyDiffuseInputs`. Also deferred by choice: §G2 (QuestionRecall nesting changes the persisted report schema — author decision), §C1's public parameter-object restructuring, and the `campaign_validation.py` pydantic merge (documented as deliberate independent re-validation).
