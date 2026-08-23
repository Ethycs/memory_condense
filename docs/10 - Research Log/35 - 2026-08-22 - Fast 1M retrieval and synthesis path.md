# Fast 1M retrieval, CAV fusion, and synthesis path

**Status:** the heavy locked-validation build campaign was intentionally stopped
after six sealed ten-question shards. The practical development benchmark now
reuses the sealed original 1,039,203-token retrieval artifact directly. A fresh
fixed-S1 run completed with ten Terra answer calls and ten independent Sol judge
calls, without rebuilding a corpus, opening a retrieval store, loading Qwen, or
rerunning S0--S3 retrieval.

This entry records the operational correction: the immediate goal is a fast
memory-retrieval-plus-summarization benchmark, not exact certification of ten
new million-token validation stores. The fresh fixed-S1 result below is a
text-only control. It does **not** yet exercise the intended fourth layer,
Pure Attention CAV Routing.

## Canonical stack

The retrieval stages are strictly cumulative. A child retains its parent's
ordered evidence and may only append evidence that fits the remaining budget.

```text
S0  causal/coverage protected baseline
└── S1  + direct anchor episodes
    └── S2  + representative/bridge episodes
        └── S3  + artifact-global closure

selected cumulative prompt
└── Pure Attention CAV Routing
    ├── K latent concepts extract from N evidence nodes  [K,N]
    └── N evidence nodes read the K concepts             [N,K]
        └── residual concept reinjection

reordered/grouped exact evidence
└── LLM synthesis/rescoring
    ├── role and density labels for episodic additions
    ├── cited, quote-grounded claims
    └── canonical final answer
```

| Layer | Artifact name | Exact meaning |
| --- | --- | --- |
| S0 | `causal_graph_coverage_predecessor` | The frozen-v3-compatible causal-graph, packing, and coverage-selected prompt. It is the protected root, not an episodic method. |
| S1 | `direct_episode_additions` | `S1 = S0 +` novel evidence from episodes containing protected anchors and configured neighboring episodes, plus direct fallbacks. |
| S2 | `representative_episode_additions` | `S2 = S1 +` novel evidence from independently routed representative episodes intended to recover bridge or diffuse evidence. |
| S3 | `artifact_global_closure_additions` | `S3 = S2 +` novel matching units discovered by artifact-wide closure over the union of direct and representative seeds. |
| Latent CAV routing/fusion | `LatentEvidenceRouter`; Pure Attention CAV Routing | The remembered fourth layer above retrieval. K learned latent concept tokens read N selected evidence-node features; the N nodes then read the filled K concepts and receive a residual update. It retrieves no new text. It replaces quadratic node-to-node graph attention with two linear cross-attention passes. |
| Synthesis/rescoring | synthesis policy v3 | A later answer layer: an LLM labels episodic evidence by semantic role and density, constructs exact-quote-cited claims, and produces the answer. It is separate from CAV reinjection. |

The defining equations are:

$$
C_1 = \operatorname{MHA}(Q=C_0, K=X, V=X),\qquad E:[K,N]
$$

$$
X_1 = X + \operatorname{MHA}(Q=X, K=C_1, V=C_1),\qquad R:[N,K]
$$

where $N$ is the selected evidence-node count and $K\ll N$ is the fixed
latent-concept count. This changes the global-routing cost from
$O(N^2D)$ to $O(NKD)$. The authoritative design source is
[`graph_transformer_cav_summary.md`](../00%20-%20Theory/graph_transformer_cav_summary.md),
and the bounded implementation contract is
[Episode-Primary Latent Evidence Fusion](../02%20-%20Implementation/04%20-%20Episode-Primary%20Latent%20Evidence%20Fusion.md).

The repository already implements the generic and same-GPU extraction /
reinjection equations, exact route-matrix receipts, and an extractive renderer.
The intended CAVs here are the per-query updated latent concepts $C_1$: the
learned $C_0$ tokens read the selected evidence and the resulting $C_1$ values
are written back into the evidence-node residuals. This is not the separate
fixed-bank operation of projecting evidence onto previously persisted CAV
directions.

The current executable path nevertheless stops short of the treatment in the
theory note. `LatentEvidenceRouter` computes $X_1$, but the generic planner only
records its hash, and the resident executor validates and then releases the
full steered-node tensor. Both paths derive extractive groups from the $[K,N]$
and $[N,K]$ attention matrices instead. Thus the surviving behavior is an
attention-routing proxy: it uses the extraction/reinjection weights to reorder
exact text, but downstream inference never consumes the reinjected node states
$X_1$ themselves. The router is also untrained and has only synthetic smoke
evidence.

Consequently, the missing fourth-layer experiment is not "compute another CAV
score" and is not, by definition, residual injection into the final answer
LLM. It is a matched post-S0--S3 arm that preserves $X_1$ and makes those
reinjected evidence-node representations affect the evidence selection or
synthesis path. Direct answer-model activation injection is a possible later
variant, but it requires an open-weight responder; the graph-transformer design
itself specifies reinjection into the evidence nodes. The fixed-S1 Terra run
below exercises neither variant and must remain the text-only control.

The synthesis role labels are `decisive`, `supporting`, `temporal_bridge`,
`qualifier_or_conflict`, `context`, `redundant`, and `irrelevant`. Its density
labels are `critical`, `high`, `medium`, `low`, and `none`. The historical local
Qwen forced-choice score is a separate numeric diagnostic; it must not be
relabeled as calibrated evidence density.

The separate LLM synthesis contract and measured density distribution remain in
[Research Log 24](24%20-%202026-08-21%20-%20LiteLLM%20Terra%20episodic%20synthesis%20and%20rescoring.md).

## Why stores were being rebuilt

The locked 100-question certification design divided validation into ten
different ten-question concatenations. Each shard has a different source
corpus and therefore a different compiled-store identity. Exact certification
requires building and sealing each store independently so that source hashes,
retrieval receipts, and held-out-query bindings cannot be mixed.

That was the wrong operational objective for the requested development test.
The original concatenated-memory artifact already contains, for every question
and stage, the exact provider messages and evidence needed for an answer run:

| Property | Reused value |
| --- | --- |
| Retrieval artifact | `eval_results/longmemeval-1m-recall-guarded-cumulative-development-20260821/retrieval.json` |
| Retrieval SHA-256 | `aa22f7c18470d9a7c931fd16f8f58bf67d8566e2298a45371ee2815c11a9bd97` |
| Concatenated transcript | 1,039,203 token proxies; 5,400 turns |
| Questions | 10 original development questions |
| Provider calls persisted by retrieval | 0 |

Reading this 2.24 MB JSON artifact replaces the 50--75 minute corpus/store
build for repeat answer experiments. The partial offset-60 validation build was
stopped and preserved when the objective changed; it is not a failed or sealed
result. The first six validation shards remain valid for the claims already
recorded in Logs 28--33, but the unfinished 100-question campaign is no longer
the active execution path.

## Measured retrieval ladder on the reused artifact

| Stage | Evidence rows | Increment | Mean evidence F1 | Mean context-token proxy | Distinct prompt changes from parent |
| --- | ---: | ---: | ---: | ---: | ---: |
| S0 | 354 | protected root | 0.149204 | 2,127.4 | n/a |
| S1 | 525 | +171 | 0.156102 | 6,538.4 | 10/10 |
| S2 | 530 | +5 | 0.156102 | 6,709.8 | 2/10 |
| S3 | 530 | +0 | 0.156102 | 6,709.8 | 0/10 |

S1--S3 therefore contain 30 logical question-stage rows but only 12 unique
provider prompts. S1 changes all ten prompts relative to S0; S2 changes only
`bbf86515` and `gpt4_7abb270c`; S3 changes none. A full S0--S3 answer comparison
contains 40 logical rows but only 22 unique prompts. Prompt-hash memoization can
project those 22 answers back onto all 40 rows without changing the linear
comparison.

The earlier Terra density run classified the five S2-only additions as
`none`/irrelevant and had no S3-only additions to classify. This is why the
practical default is S1, while S2 and S3 remain useful matched ablations rather
than compulsory work for every query.

## Fresh fast fixed-S1 run

The existing fixed-stage runner was sufficient; no new retrieval runner or
store format was needed. Provider-free preflight certified ten questions and
ten unique answer prompts before the first network call. The live run then used
the controlled LiteLLM Terra route with an 8,000-token prompt cap, 256-token
output reserve, zero retries, and a fresh output root.

This answers “does the already-retrieved 1M memory packet support the task?”
It does not answer “does latent CAV reinjection improve that packet?” That
question needs a distinct matched arm in which the computed $X_1$ node states
survive and change the downstream evidence/synthesis input. Terra can still be
the text synthesizer if the steered node states are converted into a bounded
selection or ordering policy first. Only the stronger answer-model activation
injection variant requires an open-weight responder; the remote Terra endpoint
exposes no activation hook.

| Result | Value |
| --- | ---: |
| Answer artifact | `eval_results/longmemeval-1m-fast-s1-benchmark-20260822/final-answers.json` |
| Answer artifact SHA-256 | `e9ee4705a5fcba706e4e3f38456cc3fa72dd9ea2c5dff5b31e46cf953ca63a83` |
| Terra calls | 10 physical / 10 unique; zero retries |
| Terra provider elapsed | 58.843 s total; 5.884 s mean; 5.397 s median |
| Prompt-token proxy | 68,284 total; 6,828.4 mean |
| Output-token proxy | 85 total |
| Normalized exact match | 5/10 |
| Mean token F1 | 0.786009 |
| Independent Sol judge artifact | `eval_results/longmemeval-1m-fast-s1-benchmark-20260822/semantic-judge-sol.json` |
| Judge artifact SHA-256 | `1e7ac4cd33e2c8b397fec2147f1c1073f4169c6a82b289ea2ce1dabf1a109bb3` |
| Sol calls / elapsed | 10 physical / 10 unique; 52.992 s provider elapsed; zero retries |
| Automated semantic accuracy | 9/10 |

The sole Sol negative was prediction `Close to 1300` against gold `1300`; the
judge rejected the approximation. This is a plausible adjudication false
negative, but the sealed 9/10 artifact is not overwritten. The result is a
speed and functionality diagnostic on ten development questions, not the
abandoned formal `>=95%`/minimum-100 certification claim.

The richer synthesis-v3 diagnostic on the same fixed retrieval remains the
best measured answer transformation: it produced 6/10 exact match, 0.901019
mean F1, and 10/10 independent semantic judgments at S1, S2, and S3. Its 4,096
output-token structured contract is heavier than the fixed-S1 responder, but
prompt deduplication limits it to 12 unique calls across all three stages.

## Streamlined operating rule

For iteration on this benchmark:

1. Treat the sealed original `retrieval.json` as the retrieval result and do
   not rebuild its corpus or store.
2. Choose S1 for the default fast answer test.
3. For a linear retrieval ablation, evaluate S0 through S3 cumulatively and
   deduplicate by the exact provider-message hash: 22 unique calls, not 40.
4. Treat Pure Attention CAV Routing as the fourth method: run the same selected
   evidence through the bounded K-latent extraction/reinjection arm, then render
   the same exact atoms. Do not describe ordinary text synthesis as this arm.
5. Add LLM synthesis/rescoring afterward when evidence roles, density,
   citations, or canonical answers are the object of the test: 12 unique
   S1--S3 calls, not 30.
6. Report normalized EM/F1 and independent semantic judgment as diagnostics;
   do not invoke shard reconstruction, local Qwen, campaign merging, or exact
   replay unless the experiment specifically asks for those properties.

This restores the intended linear comparison while separating retrieval,
latent CAV reinjection, and answer synthesis.
