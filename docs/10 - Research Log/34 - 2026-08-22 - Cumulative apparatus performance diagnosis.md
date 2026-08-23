# Cumulative apparatus performance diagnosis

**Status:** this is a docs-only diagnosis. The locked provider-free campaign
sealed six shards through offset 50. Its offset-60 build was later intentionally
stopped when the active objective changed from exact 100-question certification
to the fast original 1M retrieval-and-synthesis benchmark documented in
[Research Log 35](35%20-%202026-08-22%20-%20Fast%201M%20retrieval%20and%20synthesis%20path.md).
No optimization described here was applied to the six sealed shards, and this
note does not revise or reinterpret any sealed result.

The apparatus is doing materially more work than a production fixed-S1 query
would require. Some of that work is the price of a strict experimental audit;
some is an exact-but-expensive implementation choice; and some is a candidate
for a versioned policy change. Those categories must remain separate.

## Evidence established by the completed shards

The following observations are direct artifact facts or exact consequences of
the persisted offset-50 manifest and the current implementation. They are not
runtime profiles.

| Evidence | Established observation | Scope and limitation |
| --- | --- | --- |
| Compilation revisions | The completed offset-50 store has **963 snapshots for 480 sources**: two changed publications per source, one metadata-coverage publication, and two coverage-finalization publications, or `2 * 480 + 1 + 2 = 963`. | Exact for the sealed offset-50 combined store. It does not allocate elapsed time among phases. |
| Source-root hashing | The final store has 5,403 turns plus 7,984 chunks, or 13,387 source rows. Because each in-transaction snapshot reconstructs the source and graph roots, the compiler performed **12,891,681 source-row hash visits** (`13,387 * 963`), in addition to visits over the growing graph row streams. | Exact count of source rows presented to the canonical row hasher by these snapshot calls, not a disk-I/O count or a causal timing share. |
| Later-stage yield | Across the first 60 sealed questions, S2 appended evidence on **4/60** questions (`q003`, `q004`, `q035`, and `q047`). S3 appended evidence on **0/60**. S2 reported `budget_exhausted` for 56/60 questions; S3 did so for 60/60. | This describes additions under the frozen cap, not evidence relevance or answer quality. S2 added 13 rows in total; S3 added zero. |
| Whole-question elapsed time | The 60 sealed retrieval elapsed fields sum to about 11,832.224 seconds. The three largest completed values are `q058` at **595.4114252 seconds**, `q025` at about 574.619 seconds, and `q010` at about 465.185 seconds. | Only whole-question elapsed time is sealed. No persisted phase trace can assign these differences to a particular mechanism. |

The compilation counts are reconstructible from the
[offset-50 compilation manifest](../../eval_results/longmemeval-1m-recall-guarded-cumulative-validation-20260822/shards/offset-050/combined-store/combined-cumulative-store.json).
The six shard seals and their replay/audit boundary are recorded in
[Research Logs 28--33](33%20-%202026-08-22%20-%20Locked%20validation%20offset%2050%20seal.md).

## Why compilation repeats cumulative work

The diffuse compiler enumerates each source and publishes its episode batch
and discourse-link batch separately, then publishes metadata coverage and
finalizes episode and discourse coverage. See the
[compiler loop](../../src/memory_condense/eval/diffuse_compilation.py#L212-L361).
Every changed publication appends a snapshot inside the publication
transaction; coverage finalization does the same. See
[`DiscourseStore.publish`](../../src/memory_condense/persistence/discourse_store.py#L196-L266)
and
[`finalize_artifact_coverage`](../../src/memory_condense/persistence/discourse_store.py#L824-L903).

Snapshot construction recomputes two canonical content digests. The source
digest covers turns and chunks, while the graph digest covers eleven graph and
coverage row streams. Every row is serialized into canonical JSON and fed to
SHA-256. See the
[row-stream definitions and hasher](../../src/memory_condense/persistence/discourse_receipts.py#L65-L149).
The digest cache is deliberately used only when the connection is outside a
transaction, because a rollback could otherwise restore revision counters.
Snapshot publication occurs inside a transaction, so the cache cannot serve
these calls. See
[`_current_high_water`](../../src/memory_condense/persistence/discourse_receipts.py#L167-L211)
and
[`_append_snapshot`](../../src/memory_condense/persistence/discourse_receipts.py#L279-L307).

This gives two different scaling effects:

- the unchanged source stream is revisited once per snapshot, producing the
  exact 12,891,681 source-row visits above; and
- the graph stream grows after each source publication, so cumulative
  graph-row hashing approaches quadratic work in the number of publications.

Similar source counts therefore need not imply similar build times. The
rule-based linker also creates a prefix slice and reverse-scans it while
looking for prior typed units. Its worst case within a long or cue-dense
source is quadratic, so source-length skew and cue density can materially
change work at the same total row count. See
[`_nearest_prior` and `RuleBasedDiscourseLinker.link`](../../src/memory_condense/ingest/discourse_linker.py#L242-L391).

## Code-backed retrieval cost candidates

These mechanisms are present in the implementation and can produce substantial
work. They explain why the apparatus can be heavy; without phase telemetry,
they do not prove which one dominated any particular question.

### Eager cumulative-stage construction

The cumulative retrieval path builds the predecessor, direct expansion,
representative expansion, and their closures before it packs S1, S2, and S3.
Only the S3 union closure requests an artifact-global graph scan; the direct
and representative closures use seeded-graph scope. Nevertheless, all three
plans repeat overlapping work before the stage packer establishes that no
budget remains. See
[`retrieve_recall_guarded_cumulative_packet`](../../src/memory_condense/eval/_recall_guarded_cumulative_ops.py#L480-L918)
and
[`_pack_additions`](../../src/memory_condense/eval/_recall_guarded_cumulative_ops.py#L336-L387).

The frozen ladder needs those executions to measure incremental S2/S3
behavior. A production fixed-S1 request does not. The observed **S2 4/60** and
**S3 0/60** addition rates make residual-budget gating an especially important
post-campaign benchmark, but they do not by themselves authorize changing the
running experiment.

### Repeated source-universe and mutable-store validation

Episode routing scans and quote-hashes the source chunk universe, ranks every
source before applying the cap, and representative retrieval validates the
same source universe again. See
[`scan_discourse_source_chunks` and `rank_episode_source_candidates`](../../src/memory_condense/application/discourse_sources.py#L63-L202),
[`route_discourse_episode_sources`](../../src/memory_condense/application/discourse_workflow.py#L301-L346),
and
[`retrieve_discourse_episode_representatives`](../../src/memory_condense/application/discourse_workflow.py#L637-L680).

Per-source episode and representative reads also hydrate and revalidate
episode ordering. That is a real integrity rule for a generic mutable store,
not dead code. See
[`_validate_source_episode_order`](../../src/memory_condense/persistence/discourse_store.py#L353-L383),
[`get_representatives`](../../src/memory_condense/persistence/discourse_store.py#L677-L707),
[`episodes_for_source`](../../src/memory_condense/persistence/discourse_store.py#L970-L990),
and the
[test that requires read-time revalidation](../../tests/test_discourse_store.py#L286).
Any cache must therefore be keyed by an immutable snapshot/read lease or be a
validated bulk read; a process-global mutable-store bypass would be unsound.

### Representative-model fan-out and final attention width

Representative retrieval admits sources, forms groups, and calls the nested
linker after collecting candidates. See
[`retrieve_episode_representatives`](../../src/memory_condense/search/episodes/representative_retrieval.py#L540-L756).
At the campaign maxima of 256 total episodes, group size 8, beam 2, and linker
capacity 64, the nested tournament can make 32 serial group calls followed by
one final call over as many as 64 finalists. See
[`QwenMemoryLinker.inspect_nested`](../../src/memory_condense/associations/qwen_memory_linker.py#L504-L644).

The linker limits sequence tokens, but its final joint call still enters the
prefix encoder. That encoder constructs a full query-key logits tensor and
applies softmax over it, making cost sensitive to joint sequence length and
attention area. See
[`QwenMemoryLinker.link`](../../src/memory_condense/associations/qwen_memory_linker.py#L100-L254)
and
[`QwenPrefixEncoder.capture_layers`](../../src/memory_condense/modeling/qwen_prefix.py#L763-L916).

### Set/performance frontier scoring

The active-partition scan has a special venue/performance-set branch. It
streams all rows from routed sources, normalizes and classifies candidates,
then performs dense coverage scoring in sequential microbatches. Its live
report includes an `active_partition_scan_elapsed_s` field, but that phase
timing is not preserved in the sealed per-question artifacts. See
[`_scan_active_partition_frontier`](../../src/memory_condense/application/partition_workflow.py#L134-L525),
[`PrefixAdmissionSelector`](../../src/memory_condense/search/selectors/prefix_admission.py#L258-L409),
and
[`CausalChoiceScorer`](../../src/memory_condense/search/selectors/causal_choice_scorer.py#L565-L725).

`q058` is a performance-combination query, so this branch is a plausible
candidate. It is not an established cause of the 595.4114252-second elapsed
value. The sealed artifact lacks the phase timing, representative candidate
counts, model-pass counts, token shapes, GPU synchronization points, SQLite
row counts, and cache state needed to distinguish this branch from
representative fan-out, global scans, packing retries, or an interaction among
them. The only defensible q058 claim is that its sealed whole-question timing
is an outlier.

### Full-row hybrid source search and closure scans

Hybrid search streams every embedding row for the selected sources, decodes
and normalizes it in Python, and retains only the best bounded buffer. The
`candidates_per_source` limit bounds retained results, not scanned rows. See
[`hybrid_query_sources`](../../src/memory_condense/search/indexes/hybrid_queries.py#L130-L293).
The S3 artifact-global closure similarly streams and parses every route. See
[`scope_scan`](../../src/memory_condense/search/closure/scope_scan.py#L63-L160)
and
[`stream_artifact_routes`](../../src/memory_condense/persistence/discourse_queries.py#L22-L50).
Both are production-relevant implementation costs when those paths are used.

## Experimental overhead versus production cost

| Category | Current work | Intended treatment |
| --- | --- | --- |
| Campaign certification | Repeated source/target store-identity reconstruction and database/index file hashing during build and open. See the [cumulative runtime](../../src/memory_condense/eval/recall_guarded_cumulative_runtime.py#L184-L255) and its [build/open checks](../../src/memory_condense/eval/recall_guarded_cumulative_runtime.py#L425-L642). | Retain for independent campaign certification. Do not include it in a production query-latency claim. Benchmark it separately. |
| Incremental-stage experiment | Eager S2 and S3 construction even when the protected prompt has exhausted the cap. | Preserve for this frozen ablation campaign. Afterward, make production execution residual-budget driven under a new retrieval-policy identity. |
| Exact implementation overhead | Repeated source scans, per-source hydration/validation, Python full-row embedding scans, route scans, and quadratic linker behavior. | Optimize with exact output-equivalence gates; retain receipt and validation semantics. |
| Model policy cost | Representative tournament fan-out, a final call as wide as 64 candidates, and full joint attention. | Bound or replace only under a versioned policy with recall/answer gates. |
| Compilation provenance | One v1 snapshot per changed publication creates exact intermediate history but repeatedly rebuilds cumulative roots. | Constant-factor fixes may preserve v1. Publication batching changes revision history and therefore requires a v2 compilation receipt. |

## Six-step post-campaign optimization sequence

The order matters: measure first, land exact behavior-preserving changes next,
then version the changes that alter execution or identity semantics.

1. **Add identity-safe phase telemetry.** Record source scans, episode build,
   discourse linking, inserts, source-root hashing, graph-root hashing,
   coverage finalization, HNSW close, store identity, file hashes, each
   retrieval stage, model calls, input tokens/attention elements, GPU peak,
   and SQLite query/row counts. Synchronize the GPU around measured model
   phases. Put nondeterministic timing in a sidecar keyed by the semantic
   receipt/probe identity.

2. **Apply byte- and behavior-preserving implementation fixes.** Replace the
   linker's prefix slicing/reverse scans with an O(n) typed prior index;
   compile and reuse the source universe; bulk-fetch episodes,
   representatives, and evidence; cache validation only under an immutable
   snapshot lease; cache artifact routes and source partition inventory;
   cache normalized source embeddings or run an exact vectorized search; and
   split the stable committed-source digest cache from the changing graph
   digest.

3. **Stop eager production execution.** Execute predecessor, direct/S1
   expansion, closure, and packing first. Inspect the residual token budget
   and unmet obligations before optionally executing S2, then do the same
   before S3. A fixed-S1 production path must make zero representative-linker
   and artifact-global-closure calls. A skipped-stage receipt must state that
   the stage was not executed; it must not claim an evaluated ablation or
   silently reuse `budget_exhausted` semantics.

4. **Bound or replace the representative tournament.** First benchmark a
   separate representative linker capped at eight candidates while sharing
   the encoder. Then compare independent-row QK/OV batching plus a small final
   tournament. Enforce an attention-element ceiling rather than only a token
   ceiling. These are policy changes and require a new identity.

5. **Add deterministic structural fast paths.** When typed coverage already
   proves the required frontier, bypass neural scoring for those proven
   reservations. Longer term, compile a typed occurrence index with explicit
   per-chunk coverage and no-output receipts so the fast path remains
   independently auditable.

6. **Batch compiler publications.** Exact v1 work can reduce constants while
   retaining all 963 revisions and their roots. A v2 compiler can instead use
   an outer transaction or deterministic source batches, retain ordered
   per-source receipts, finalize coverage, and append one final snapshot per
   batch. If intermediate snapshots remain a requirement, benchmark a Merkle
   construction rather than reconstructing all cumulative row streams.

## Receipt and identity boundary for v2

Timing must not be part of semantic identity. Today
[`CoverageSelectionReport`](../../src/memory_condense/search/selectors/coverage_models.py#L120-L138)
contains `elapsed_s`, and the predecessor receipt hashes that report. See the
[coverage-report receipt binding](../../src/memory_condense/eval/_recall_guarded_cumulative_ops.py#L300-L321).
A v2 semantic projection should exclude elapsed fields and link a separate
performance sidecar by semantic receipt SHA-256 and probe ID. This is a
forward change only; existing v1 hashes remain authoritative for their
artifacts.

The remaining proposed identity changes are explicit:

- lazy S2/S3 execution, new `not_executed` receipt semantics, representative
  caps, attention ceilings, and structural fast paths require a v2 retrieval
  policy/implementation identity;
- batched publication changes graph revision history and final snapshot
  SHA-256 even if all final domain rows and content roots match, so it requires
  a v2 compilation format and receipt;
- independently reconstructible caches and bulk reads may retain the v1
  receipt format and policy semantics only if they reproduce byte-identical
  units, relations, seeds, evidence, ordering, content roots, and semantic
  receipt projections. Changed code still receives a new implementation
  SHA-256.

No v2 result may be merged into, substituted for, or used to retroactively
reinterpret the frozen v1 campaign.

## Benchmark gates

No optimization is accepted on a wall-clock improvement alone.

1. **Compiler scaling gate:** run 10k, 25k, and 50k chunk builds with balanced
   and long-source-skewed layouts at 0%, 25%, and 100% discourse-cue density.
   Report each telemetry phase, total snapshots, row-hash visits, and scaling
   slope.

2. **Exact-change gate:** for every v1-compatible change, require identical
   units, relations, seeds, evidence, ordering, final source/graph content
   roots, coverage receipts, and semantic receipt projections.

3. **Compiler-v2 provenance gate:** require identical final domain tables
   except revision history, identical final source and graph content roots,
   identical coverage receipts, independently reconstructed roots, and the
   declared deterministic snapshot/batch count. Scaling should approach
   linear rather than merely meeting an arbitrary latency target.

4. **Retrieval performance gate:** benchmark `q058`, `q054`, `q025`, the
   campaign median, and both set/performance and non-set queries, cold and
   warm, for at least five repetitions. Report p50, p95, maximum, model calls,
   tokens/attention elements, CUDA peak, SQLite queries/rows, and source-scan
   counts. This is the test that can investigate q058 causality; the current
   artifact cannot.

5. **Execution-boundary gate:** instrument fixed S1 and assert zero
   representative-linker calls and zero artifact-global-closure calls. Verify
   that every skipped-stage receipt states `not_executed` and makes no
   incremental-stage effectiveness claim.

6. **Quality gate:** compare evidence/source recall and downstream fixed-S1
   answer accuracy, not latency alone. Any policy-changing representative or
   structural fast path must pass the same held-out answer boundary before it
   can replace the frozen treatment.

## Campaign boundary and next decision

No optimization in this note was applied to offset 60 or any prior shard. The
offset-60 process was stopped before sealing, and offsets 70--90 were not
started. The deterministic 100-question merge, `>=95%` gate, and
production-bound Mem0 comparison therefore remain incomplete by design.

The active fast path reuses the original sealed retrieval artifact and does
not need these campaign optimizations. If exact certification resumes, step 1
should land first on a new benchmark branch. The exact v1-compatible work in
step 2 can then establish a trusted baseline. Steps 3--6 proceed only with
their declared v2 identities and independent reconstruction gates.
