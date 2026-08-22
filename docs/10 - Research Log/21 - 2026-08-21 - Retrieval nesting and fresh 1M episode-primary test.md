# Retrieval nesting and fresh 1M episode-primary test

**Status:** architecture map plus a controlled, provider-free, real-model
functional result. The fresh million-token run passed ingest, restart,
compilation, route execution, and receipt verification. It failed retrieval
quality: literal answer reachability fell to 3/10, and the compact summary
reported zero gold-source recall. The latter figure still needs a retained-ID
audit. This is **not** an accuracy pass, a 95% result, an EM-LLM result, or a
Mem0 comparison.

**Primary artifact:**
`eval_results/longmemeval-1m-episode-primary-controlled-20260821/result.json`.
The verified route phase identity is
`8f09948145ddf0d56371c64762c3db1d44d7128f41fa9dbfb97b7d77c78e5ce6`.

## Result in one sentence

The retrieval methods are not one flat list: chunk, span, source, graph, and
packed-memory modes choose or extend initial evidence; diffuse episodic
retrieval sits above one of those anchor methods; and `episode_primary` uses
the anchor result only to route sources before Qwen selects the episode seeds
that enter bounded graph closure. That complete stack ran over a newly built
1,041,276-token memory, but it recovered 0/10 gold evidence sources and only
3/10 literal answers. The literal result is directly auditable from the saved
summary; the source result is reported but not independently reproducible from
that summary because it omitted the expected and retrieved source-ID sets.

## The three independent axes

The word *retrieval* currently names three different choices. They should not
be compared as though they were siblings.

1. **Stored representation:** raw turns become chunks and indexes; optional
   write-time processes also create memory items, causal associations, or a
   compiled episode/discourse graph.
2. **Initial query method:** one `RetrievalConfig.mode` selects chunks, spans,
   sources, a hybrid graph result, or an already packed memory context.
3. **Diffuse episodic route:** after an initial anchor method has run, either
   `legacy_union` or `episode_primary` chooses the seeds supplied to discourse
   closure and the atomic evidence packer.

The episode segmentation arms (`fixed_interval`, `lexical_embedding`, and
`qwen_head`) belong to the first axis. They are alternative graph builders,
not three query-time retrieval modes.

## Complete nesting map

```text
ingested turns
└── exact source-backed chunks
    ├── BGE vectors + HNSW
    ├── BM25 term index
    ├── span/source pools
    ├── optional extracted memory items
    ├── optional learned causal associations
    └── optional diffuse compilation
        ├── fixed_interval boundaries
        ├── lexical_embedding boundaries
        └── qwen_head boundaries
            └── episodes → discourse units → discourse relations

question
└── choose one initial query method
    ├── dense
    ├── hybrid
    ├── span
    ├── source
    ├── anchored_source      = hybrid → complete activated sources
    ├── hybrid_source        = hybrid → bounded activated-source search
    ├── hybrid_neighbor      = hybrid → source-local temporal neighbors
    ├── hybrid_graph         = hybrid → routed graph/source expansions
    ├── memory               = memory items + hybrid chunks → ContextPacker
    ├── causal_consolidation = hybrid chunks → causal expansion → ContextPacker
    └── causal_graph         = hybrid_graph → causal expansion → ContextPacker

diffuse analysis only
└── initial anchors from the selected query method
    └── source-candidate scope
        ├── legacy_union
        │   └── anchor episodes + direct chunk fallbacks
        │       + optional representative episodes
        └── episode_primary
            └── representative episodes only; no direct chunk fallback
                └── seeded discourse closure
                    └── atomic evidence packet
                        └── final QA prompt
```

The important non-nesting is that `memory`, `causal_*`, and
`episode_primary` are not progressively enabled by selecting a single larger
mode. The ordinary benchmark chooses one top-level `RetrievalConfig.mode`.
The diffuse analysis path separately uses a direct mode to acquire anchors,
then applies an episodic route over the compiled discourse artifact.

## What each `RetrievalConfig.mode` actually wraps

| Mode | Immediate substrate or parent | What it returns or adds |
| --- | --- | --- |
| `dense` | BGE query vector + HNSW | Top-*k* chunks by cosine similarity. This is the flat baseline. |
| `hybrid` | Dense candidates + BM25 candidates | A blended, reranked chunk list. The legacy `hybrid=true` flag is an alias for this effective branch. |
| `span` | Pooled vectors over contiguous chunk spans | Member chunks from the winning token-sized spans. It is a parallel base method, not hybrid plus neighbors. |
| `source` | Pooled source/session vectors | Complete selected source groups by dense source similarity. |
| `anchored_source` | `hybrid` | Hybrid anchors activate source IDs; the method hydrates and fairly interleaves the complete activated sources. |
| `hybrid_source` | `hybrid` | Preserves the top-*k* anchors and adds bounded candidates from sources represented in an independently bounded activation prefix. Optional source-local search reranks inside those sources. |
| `hybrid_neighbor` | `hybrid` | Adds bounded previous/next chunks from the anchors' own sources. Replacement slots may trade weak anchors for neighbors. |
| `hybrid_graph` | `hybrid` | Preserves hybrid anchors, then composes optional partition routing, role weighting, query facets, temporal neighbors, source activation, source-local search, TF-ISF, hierarchical source contraction, and Qwen rerank or feedback. |
| `memory` | `build_context` | In the benchmark, retrieves extracted memory items plus hybrid verbatim expansions and sends them through `ContextPacker`; recent haystack turns and live consolidation are disabled. |
| `causal_consolidation` | Hybrid expansions inside `build_context` | Adds bounded candidates reached through the learned causal/co-activation graph, then packs the result. Benchmark probes do not learn from one another. |
| `causal_graph` | `hybrid_graph` followed by `build_context` | Supplies the graph result as the direct expansion set, applies learned causal expansion, then packs it. This is the deepest ordinary benchmark mode. |

Every ordinary benchmark branch eventually becomes bounded context and the
same QA prompt. The chunk-returning modes are capped after converting
`RetrievalResult` values to text. The memory and causal modes use
`ContextPacker` before the common final prompt cap.

## Modifiers that are not additional top-level modes

Several controls alter a branch without creating another level in the mode
list:

- role-aware ranking, multi-fact source diversity, query-facet reservation,
  source partition routing, TF-ISF activation, hierarchical source
  contraction, and source-local search live inside `hybrid_graph` and the
  graph portion of `causal_graph`;
- Qwen source reranking and Qwen activation feedback are mutually exclusive
  modifiers of source-local graph retrieval;
- coverage selection is a post-retrieval selector available to packed memory
  or causal modes; and
- `search_associative` and `expand_hebbian` are application APIs rather than
  current `RetrievalConfig.mode` values. The former starts with hybrid anchors
  and spends bounded slots on stored QK/CAV links; the latter expands already
  acquired anchors through live co-access edges.

These Qwen source controls are also distinct from Qwen episode
representative selection. One reranks chunks inside activated sources; the
other selects episode seeds after the source-candidate scope has been sealed.

## The two diffuse episodic routes

Both routes start after diffuse compilation and after the initial retrieval
method has produced exact chunk anchors.

### `legacy_union`

1. Direct anchors are mapped to nearby compiled episodes.
2. Previous and next episode windows may be added under the episode policy.
3. Chunks that cannot be mapped may survive as bounded direct fallbacks.
4. If representative retrieval is supplied, its episode seeds are unioned
   with the direct seeds.
5. The combined episodes and direct chunk IDs enter `artifact_global`
   discourse closure, which may also scan matching units across the compiled
   artifact before relation expansion and packing.

This is the compatibility path. Direct chunk retrieval can influence closure
and the final packet without first winning Qwen episode selection.

### `episode_primary`

1. The initial anchors help build a bounded, independently receipted source
   candidate scope.
2. Qwen inspects bounded episode representatives inside that scope.
3. Only the selected representative episodes become closure seeds.
4. The direct-expansion plan is explicitly empty: zero direct episode seeds
   and zero direct chunk fallbacks.
5. Closure is forced to `routing_scope=seeded_graph` before atomic packing.

Thus `episode_primary` does not eliminate the initial chunk method. It changes
the authority of its output: anchors route the episode search, but they do not
enter closure as evidence seeds. Route v2 is a structural certification
wrapper around this execution; it does not perform an additional retrieval.

## Exact stack exercised by the fresh 1M run

The test rebuilt the original locked validation concatenation in a new store.
No old compiled or causal database was reused.

```text
1,041,276-token locked concatenation
→ deterministic ingest (5,551 turns)
→ current 120–250-token chunker (8,122 chunks)
→ BGE-M3 vectors and HNSW
→ close and reopen the store
→ fixed-interval diffuse compilation
  (1,119 episodes, 7,623 units, 8,873 relations)
→ hybrid_graph anchors
  → four-way partition routing
  → dense/BM25 pool and ten retained anchors
  → role-aware ranking
  → forward temporal neighbors
  → TF-ISF and two-hop hierarchical source activation
  → bounded source-local search
→ episode source scope = RRF(anchor sources, all-source TF-ISF), at most 64 sources
→ Qwen representative retrieval, top eight episodes
→ episode_primary seeded closure, at most three hops
→ atomic packet, at most 7,000 context-token proxies
→ route-v2 verification
```

Query-facet reservation, multi-fact diversity, legacy Qwen chunk reranking,
and Qwen activation feedback were disabled in this run. The initial
`hybrid_graph` result and the later Qwen representative pass were therefore
two separate retrieval stages, not duplicate names for the same operation.

## Functional result

The local checkpoint identities were:

| Runtime | Checkpoint SHA-256 |
| --- | --- |
| BGE-M3 | `a3d5c49f064ab58d7cf5bba1c2085918f529778e88535aca7de674c9094af0b7` |
| Qwen3-8B prefix | `76273516aa6924b12344d5e83daa485b66459b663c745cb3b9ef51cc17c7440d` |

The new store closed, persisted its HNSW index, and reopened with all 5,551
turns before episode compilation. All ten queries then returned verified
`episode_primary` records with exactly eight representative seeds each.

| Measure | Result |
| --- | ---: |
| Transcript tokens | 1,041,276 |
| Turns | 5,551 |
| Current chunks | 8,122 |
| Fixed-interval episodes | 1,119 |
| Discourse units | 7,623 |
| Discourse relations | 8,873 |
| Questions | 10 |
| Literal answer present in packet | 3/10 |
| Any complete gold source set | 0/10 |
| Reported mean gold evidence-source recall | 0.0 |
| Closures claiming completion | 0/10 |
| `workspace_cap` closures | 8/10 |
| `conflicted` closures | 2/10 |
| Maximum packet context-token proxy | 6,989 |
| Elapsed wall time | 3,032.718 seconds |

Gold answers and evidence labels were used only after the complete route phase
had returned and verified. There was no responder, judge, or remote provider
call. The functional status is therefore a plumbing result: real local models,
fresh million-token storage, restart, compilation, ten route executions, and
sealed receipts all worked. It is not a retrieval-quality pass.

## Why this is worse than the earlier 1M result

This run did **not** recreate the earlier concatenated-memory treatment. It
reused the million-token experiment shape while changing both the population
and the retrieval authority. The earlier headline result used a selected
development concatenation and the frozen v3 `causal_graph` treatment. This run
used validation offset 0 and replaced that treatment with fixed-interval
`episode_primary` retrieval.

That change was experimental scope drift. The run was configured to answer
whether the newly implemented episodic route could execute at one-million-token
scale, when the requested control was a recreation of the original
concatenated-memory test. It is valid as an `episode_primary` functional
ablation, but it must not be presented as a replay of the original treatment.

The closest control is not the selected development pilot. Frozen v3 was
already audited on the exact validation offset-0 population used here. That
gives the following same-population comparison:

| Run | Retrieval authority | Literal answer in final context | Mean labeled source recall | Every labeled source | Mean final context |
| --- | --- | ---: | ---: | ---: | ---: |
| Frozen v3, validation offset 0 | `causal_graph`: direct hybrid-graph evidence, learned causal expansion, coverage selection, and selected-partition scanning | 6/10 | 93.334% | 8/10 | 2,205.9 tokens |
| Fresh current run | `episode_primary`: anchors route sources, then eight representative episodes alone seed bounded closure | 3/10 | 0.0% reported | 0/10 reported | 5,220.6 tokens |

The v3 row is recomputed from
`eval_results/validation-v3-offline-recall-offset-000.csv`; the current row is
from the primary artifact named above.

The 6/10 to 3/10 literal drop is therefore a genuine final-context regression
on these ten questions, not just a comparison between an easy development
slice and a harder validation slice. It is not evidence that ingest,
persistence, BGE, or the million-token store stopped working: those gates
passed, and the current route still found three literal answers. It is evidence
that the new evidence-admission route does not yet preserve the old
treatment's final-context behavior.

The architectural change removed several recall safety nets at once:

1. v3 let direct hybrid-graph chunks enter the final treatment and then added
   learned causal expansions. `episode_primary` records those anchors only for
   source routing and supplies zero direct chunk fallbacks to closure.
2. v3's Qwen coverage selector operated over the retrieved chunk frontier and
   could scan every content chunk in the selected partitions. The new route
   reduces as many as 256 candidate episodes to eight representative seeds.
3. The compatibility route can use `artifact_global` matching-unit discovery.
   `episode_primary` forces `seeded_graph`, so a missed representative cannot
   be recovered by an artifact-wide scan.
4. All ten new closures stopped incomplete (`workspace_cap` or `conflicted`).
   They consumed 2.37x the old mean context while literal hits halved. That is
   consistent with distractor expansion rather than a tight final token budget,
   although the missing stage traces prevent a causal assignment.

There are secondary implementation differences too: current chronological
ingest and chunking produced 8,122 chunks rather than v3's 8,128, and the new
graph contains 1,119 fixed-interval episodes rather than v3's learned causal
staging. Those are confounders, but the aggregate result cannot assign the
loss to one of them.

The test artifact also has a diagnostic limitation: it kept the final metrics
and durable store but not the complete per-question route phase, expected
source IDs, retrieved source IDs, or packet atoms. The durable store contains
all 18 labeled source IDs under the correct `question_id::source_id` namespace,
and every labeled source survives into episode, discourse-unit, and relation
evidence. The production metric compares those namespaced IDs verbatim, but
the compact artifact cannot prove that the ad hoc caller supplied the composed
labels. A namespace error is unlikely: composition uses the same function for
turn and evidence labels, and the three literal hits are the generic numeric
strings `$5`, `23`, and `3`, all of which can occur in distractors. The
all-zero source result nevertheless remains provisional until a replay retains
both ID sets and excludes the mismatch directly.

Even after that metric caveat, the literal result proves a final-packet
regression. It cannot show whether the first loss occurred in hybrid anchors,
the 64-source scope, episode availability, top-eight selection, closure
traversal, or packing. A faithful correction is an A/B replay on one frozen
fresh population: first the original v3 `causal_graph` treatment, then
`legacy_union`, then `episode_primary`, with the same questions, budgets,
checkpoint identities, retained source IDs, and post-hoc stage funnel.

## What the failure does and does not localize

If the reported zero source recall is reproduced with retained, correctly
namespaced ID sets, it proves that the correct sources did not survive into
the selected packets. It still would not identify the earliest stage that lost
them. A source may have been absent from the initial hybrid graph frontier,
absent from the 64-source scope, present but represented by the wrong episodes,
rejected from the top eight episode seeds, lost during closure, or omitted by
the atomic packet budget.

All ten closures were incomplete, so bounded closure is an observed pressure
point. It cannot yet be called the sole cause: the correct evidence may have
been lost before closure began.

The next diagnostic should record gold-blind retrieval artifacts first, then
score these six frozen checkpoints post hoc:

1. gold source in the initial anchor/pool result;
2. gold source in the routed source-candidate scope;
3. gold episode available inside that source;
4. gold episode among the eight representative seeds;
5. gold evidence visited by closure; and
6. gold evidence selected by the atomic packet.

That stage-wise funnel will show which nested method needs work without
changing the prompt cap or conflating storage, routing, episode selection,
closure, and packing.

The recall-guarded cumulative implementation requested after this diagnosis is
documented in [Research Log 22](22%20-%202026-08-21%20-%20Recall-guarded%20cumulative%20retrieval.md).
It protects the exact frozen-v3 rendered packet first and packs only novel
episodic/discourse evidence into the remaining budget. It is implemented,
focused-test verified, and now measured on the original 1,039,203-token
development concatenation. That result is a separate same-population S0-to-S3
campaign; it does not relabel this validation-offset-0 `episode_primary`
ablation.

## Artifact and scratch hygiene

The two incompatible legacy-cache attempts were removed. Their stores used
legacy turn IDs, pre-chronology timestamps/order, and obsolete chunk spans, so
they could not represent the current concatenation. The successful fresh
store and its concise result remain under
`eval_results/longmemeval-1m-episode-primary-controlled-20260821/`. All
temporary `.tmp-*` test directories were removed after the store closed.
