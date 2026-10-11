# Native experimental modules

**Status:** Integrated behind opt-in profiles; baseline defaults retained after quality assay  
**Date:** 2026-10-09  
**Applies to:** Native proxy recall and published IO learning

The seven dormant backend mechanisms from the October inventory have native
summary-based adapters. An eighth arm, precomputed CAV reinjection, tests
the original low-rank compute hypothesis. A ninth arm tests sparse spectral
propagation over BGE summary addresses. The shipping cap-8, automatic BGE tactic selection,
hierarchy context, decision-history annotations and neighborhood policy remain
the baseline. Experimental modules supplement that path through one policy and
one exact-source hydration boundary.

## Module inventory

| Profile name | Current integration | Model work |
| --- | --- | --- |
| `learned_associations` | Reads the existing IO consolidation graph, projecting original chunk links onto current atomic summary addresses. | None at recall. |
| `heat_diffusion` | Uses the existing conserved scalar heat engine over those published co-access links. This adapter uses co-access conductivities, not legacy QK/OV artifact edges. | None at recall. |
| `qwen_summary_routing` | Runs the existing summary-attention tournament on a bounded forest of BGE-shortlisted atomic summaries. This is an additional recall arm, not an unbounded traversal of the whole corpus. | Two bounded prefix passes. |
| `qwen_associations` | Uses the original Qwen linker to score question-to-summary QK associations. | One prefix pass. |
| `cav_associations` | Adds the original CAV concept-margin signal to that summary-linker score, using a checkpoint-bound concept bank. | One prefix pass; supplied CAV bundle required. |
| `cav_reinjection` | Compiles Qwen summary residuals once into rectangular CAV extraction/reinjection matrices, then reranks the existing BGE shortlist through those matrices. Qwen residuals are discarded. | Zero query-time Qwen passes; two CPU low-rank matrix-vector reads. |
| `spectral_reinjection` | Compiles a sparse degree-8 graph over resident BGE summary addresses and applies two restart-preserving propagation steps while retaining direct semantic mass. | Zero query-time model passes; sparse CPU propagation. |
| `live_head_memory` | Builds a temporary summary-only K/V workspace and performs recurrent head addressing. Discards it after the query. | At most eight summary writes plus one query capture. |
| `qwen_consolidation` | Optionally weights the existing native IO update with QK/OV activations from stored input and delivered-source summaries. Updates the same scalar graph once. | Background learning; one bounded prefix pass. |

Neighborhood tuning and forced hierarchy/history retrieval were already native
features. Continue using `--memory-dialogue-config` and
`--memory-retrieval-policy` for them. Historical evaluation arms in `tools/`
are comparison harnesses, not additional proxy modules. Full generative Qwen
reasoning remains a separate provider-based research path; the truncated local
attention model does not become a generative model through this integration.

The matched quality characterization in [Research Log 266](../10%20-%20Research%20Log/266%20-%202026-10-08%20-%20Native%20experimental%20technique%20characterization.md)
does not support enabling any module globally. The current default memory route
scored 75.42% macro over twelve engineering-history questions. CAV was the
strongest active selector at 70.14% after score audit; the combined arm scored
66.88%. Learned
links need a mature observed graph, and one-pass Qwen association remains the
best candidate for a gap-triggered retry.

## Contract and lifecycle

1. Capture a bounded immutable scalar graph when publishing the native index.
   A successful learning update publishes a new view. Active readers keep their
   prior graph and raw prefix. No graph SQL or raw transcript reads occur while
   routing a query.
2. Reuse the existing BGE query vector. Qwen receives the question and stored
   summaries under opaque aliases; it receives no raw transcript, source ID or
   span coordinate as model input. The candidate shortlist preserves the native
   dated-source scope.
3. Preserve the baseline routes and hydration. Apply separate experimental caps
   to additional whole source sections. A baseline section rejected by packing
   cannot consume the extra budget through an experimental retry. Corrupt,
   stale, overlapping or oversized extra evidence is rejected.
4. Keep original source pointers on every delivered addition. Mark additions
   `learning_eligible=false`, so graph/attention-derived evidence cannot
   reinforce itself. Ordinary independent baseline evidence keeps its existing
   learning behavior. Recalled copies retain their original ancestry.
5. Record policy, model/checkpoint/layer, real prefix-call count, shortlist
   omissions, proposed and delivered IDs, hydration diagnostics and graph
   truncation in the recall journal. Aggregate query Qwen calls appear in
   `retrieval.query_qwen_passes`; the unchanged baseline route still records its
   own zero Qwen passes. A model failure is explicit and preserves the baseline.

The adapter shares the resident compressed FP16 Qwen prefix, CUDA stream and
recall-priority GPU lock. Query-time residual experiments default to zero-based
layer **4**, which has a complete residual. Production ingest coverage continues
using layer **5**, whose unused MLP and final normalization remain absent. Capture
now stops after the requested complete layer instead of executing the unused
tail. No second Qwen model is loaded, and no per-query model transfers occur.

The graph snapshot is bounded to 8,192 qualifying stored edges by default and
reports recency truncation. It reuses existing decay, hub normalization and
observation thresholds. Refreshing this scalar snapshot does not rebuild BGE
vectors, the hierarchy or the raw index. Publication cost on a heavily learned
5M-token live session has not yet been measured for this optional snapshot.

## Enabling and rollback

Create a JSON profile or use the checkout's
[`learned.json`](../../examples/native_experiments/learned.json) or
[`attention.json`](../../examples/native_experiments/attention.json):

```powershell
memory-condense proxy --memory-experiment-config ./experiments.json
```

The regular provider, assets and data-directory options still apply. Automatic
tactic selection remains the default. The profile is validated before model
startup, applies to every session in that proxy process, and is exposed in its
retrieval status. Only `augment` mode accepts experiments. Omit the argument to
restore the normal pipeline; no store migration or destructive cleanup is needed.

Defaults are 16 shortlisted atomic summaries, up to eight in a single Qwen
workspace, four proposed additions, 1,024 additional rendered section tokens,
eight additional raw spans, and a 1,024-Qwen-token workspace. Whole summaries
that cannot fit are omitted and counted. The default recall budget is 12 prefix
passes; the provided attention profile allows 16. Learning has its own bounded
call allowance. These are recall/learning limits, not an answer-token limit.

To use CAVs, include `cav_associations` or `cav_reinjection` and set `cav_bundle` to a JSON file with
`checkpoint_sha256`, `layer`, `report`, `report_sha256`, `vectors` and
`vectors_sha256`. Report/vector paths resolve relative to the bundle. The report
must bind the training checkpoint, and its layer must match `qwen_layer`.
Artifacts are checked against the resident checkpoint before use. Old unbound
CAV artifacts are rejected. No new concept bank is silently shipped or activated.

`cav_reinjection` content-addresses each stored summary feature by checkpoint,
layer and exact summary text. A cold archive therefore pays one layer-4 Qwen
feature pass per batch once; later publications encode only new summaries. The
publication retains only `[K,N]` extraction and `[N,K]` reinjection matrices.
Recall reuses the normal BGE query, seeds the compiled concept state from the
baseline routes, and reranks only the normal BGE shortlist. It fails closed on
historical as-of scopes that differ from the compiled publication, preventing a
later summary from influencing an earlier query through the concept pool.

The October 9 completed assay compiled 783 atomic summaries through two
concepts. Cold feature construction took 167.93 seconds and 98 Qwen batches;
the cached rebuild took 2.85 seconds with zero Qwen batches. Retained state was
12,528 bytes. The query readout averaged 0.387 ms and 0.00783% of the cells in a
783-by-783 dense map. Full recall averaged 1.070 seconds versus a paired 1.020
seconds for the baseline because added raw spans still require packing. Exact
support coverage was unchanged. This removes the query-time Qwen cost from the
CAV arm; it does not make the already-Qwen-free BGE baseline faster. Audited
matched Sol scoring was 69.65% macro (54/84, zero wins, eight ties and four
losses) against the default route's 75.42% (60/84). The prior query-time CAV arm
scored 70.14%. Precomputation therefore preserved approximately the old CAV
quality while eliminating its query Qwen pass, but it did not close the quality
gap to the release baseline. See Research Log 267.

The follow-up five-concept arm retained 31,320 bytes and averaged 0.346 ms for
its low-rank readout. Ten of twelve delivered section lists were identical to
CAV-2, and it recovered no new checkpoint support. The sparse spectral arm
retained 50,112 bytes, built in 0.071 seconds at 783 nodes and averaged 0.919 ms
for its readout. It changed nine lists but also recovered no new checkpoint
support. Its exact blocked all-pairs publication builder is an assay
implementation, not a scalable million-node graph constructor. Both remain
off by default; see Research Log 268.

The repository assay can refit the existing small concept dataset with
`--gpu --fit-cav`; this produces a compatible bundle using the actual resident
weights and precision. Its concept-classification score is not memory accuracy.

## Evidence and promotion

[Research Log 265](../10%20-%20Research%20Log/265%20-%202026-10-08%20-%20Native%20experimental%20module%20integration.md)
records the real-model runs. On the saved 775,741-token engineering archive,
15 questions across seven recall configurations preserved baseline evidence in
all 105 comparisons. Combined warm retrieval averaged 1.258 seconds against
0.203 seconds for cap-8 and added 536 rendered section tokens on average.

This establishes integration, exact hydration and bounded local execution. It
does not establish improved answer accuracy or unchanged end-to-end generation
latency. Extra context can distract a reader despite preserving its prior
evidence. Keep release defaults off until a matched reader comparison supports
promotion. The small learned graph added no new delivered evidence in this
assay; its functional tests demonstrate expansion when supporting links lead
outside the baseline.
