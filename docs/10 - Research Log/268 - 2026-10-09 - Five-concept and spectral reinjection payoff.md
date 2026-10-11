# Five-concept and spectral reinjection payoff

**Date:** 2026-10-09  
**Status:** Complete local screen; five-concept and sparse-spectral arms not promoted

## Question

Research Log 267 established that precomputed CAV reinjection removes the
query-time Qwen pass, but its two broad concepts did not improve answer quality.
This follow-up asks whether a wider concept bank or a graph-frequency method can
produce a material quality gain while keeping the speed advantage. The target
was a 10% relative answer-quality improvement, with an explicit latency payoff.

## Five concepts

The first controlled dataset tried five narrow engineering-memory directions:

1. temporal revision;
2. failure diagnosis;
3. decision rationale;
4. implementation state; and
5. measurement result.

It contains 160 synthetic examples and no benchmark questions, answers or
transcript excerpts. Layer-4 FP16 Qwen separated decision rationale and
measurement result well, but failure diagnosis and implementation state failed
the existing 75% held-out balanced-accuracy gate. Those failed vectors were not
used in the retrieval arm.

The admitted five-concept bank combines the already qualified
`context_dependency` and `binding_constraint` vectors with temporal revision,
decision rationale and measurement result. Its 176 examples produced:

| Concept | Held-out balanced accuracy |
| --- | ---: |
| Context dependency | 93.75% |
| Binding constraint | 75.00% |
| Temporal revision | 75.00% |
| Decision rationale | 100.00% |
| Measurement result | 91.67% |

The resident fit itself took 2.82 seconds. The concept count does not change the
Qwen summary-feature backfill because the same cached 4,096-wide layer-4 feature
is reused for every concept.

On the same 783-summary engineering archive, five concepts retained 31,320
bytes and required 120 scalar interaction cells per query. The readout averaged
0.346 ms, compared with 0.387 ms for the two-concept arm; this difference is
timer noise, while both are safely below one millisecond. Ten of twelve delivered
section lists were identical to the two-concept arm. One changed only order and
one dropped a section. Exact evidence remained 35/92 support quotes and 11/42
complete checkpoints. Reader calls were therefore stopped: the wider bank had
created no new scored evidence to evaluate.

The learned vectors were not fully independent. Binding constraint, temporal
revision and decision rationale had pairwise cosine similarities from 0.542 to
0.813, and the five-vector participation-ratio effective rank was 3.21. A
follow-up matrix control replaced the named vectors with an orthonormal SVD
basis spanning exactly the same CAV subspace. This changed five of twelve
section lists, retained the same 31,320 bytes, and averaged 0.368 ms per
readout. It still recovered zero new support quotes. Simple concept-vector
correlation therefore does not explain the missing evidence gain.

## Sparse spectral arm

Direct graph Fourier eigendecomposition would retain a dense basis and pay an
unnecessary quadratic query cost. The implemented experimental arm instead
uses the polynomial-filter equivalent:

1. compile a directed degree-8 nearest-neighbor graph from the already resident
   normalized BGE summary addresses;
2. seed it with the protected baseline routes;
3. run two propagation steps with a 0.7 personalized restart;
4. retain 0.7 direct BGE mass and mix 0.3 propagated mass; and
5. hydrate selected summaries through the existing exact-source boundary.

The query path stores only neighbor row indices and FP32 edge weights. It runs
no Qwen pass, reads no raw content while routing, retains no transformer state,
and fails closed for a historical source scope that differs from the compiled
publication.

The current exact graph builder uses blocked all-pairs similarity during
publication. For 783 summaries this is only 613,089 similarity cells and took
0.071 seconds, but that construction is not suitable for million-node summary
sets. Production promotion would require an incremental or approximate kNN
builder. The retained graph itself is sparse: 6,264 edges and 50,112 bytes.

The spectral readout averaged 0.919 ms over 12 questions. Full recall added
21.99 ms over its paired baseline, compared with 49.80 ms for CAV-2 and 50.40
ms for CAV-5. It changed nine section lists and reached a mean Jaccard overlap
of 0.633 against CAV-2. That diversity still recovered exactly zero new scored
support quotes: total evidence stayed 35/92 and 11/42 complete checkpoints.
Reader calls were stopped for the same reason as the five-concept arm.

## Payoff matrix

Times are local recall measurements. Reader quality is reported only where an
audited matched Sol answer exists. CAV-5 and spectral values are structural
screens, not inferred answer scores.

| Arm | Audited answer macro | Query Qwen | Selector | Full recall | Publication/build | Retained state | New support over baseline | Decision |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| Protected default route | 75.42% | 0 | Included in baseline | 1.000 s | Existing BGE index | No experimental state | Reference | Keep default |
| Query-time CAV-2 | 70.14% | 1 pass | Included in Qwen pass | 1.155 s | None | Transient | 0 quotes | Reject always-on use |
| Precomputed CAV-2 | 69.65% | 0 | 0.387 ms | 1.070 s | 167.93 s cold; 2.85 s cached | 12,528 B | 0 quotes | Keep experimental |
| Precomputed CAV-5 | Not run | 0 | 0.346 ms | 0.934 s; paired baseline 0.884 s | Same feature backfill; 10.38 s observed cold-file assembly | 31,320 B | 0 quotes | Stop before reader |
| Orthogonal CAV-5 span | Not run | 0 | 0.368 ms | 0.949 s; paired baseline 0.903 s | 2.61 s cached assembly | 31,320 B | 0 quotes | Correlation control failed |
| Sparse spectral/PPR | Not run | 0 | 0.919 ms | 0.924 s; paired baseline 0.902 s | 0.071 s exact build at N=783 | 50,112 B | 0 quotes | Keep experimental; replace exact builder before scale |

The query-time speed question is answered positively but narrowly. CAV
precomputation saves about 85.75 ms per recall relative to the prior query-time
CAV arm and five concepts do not consume that saving. The 167.93-second cold
feature backfill breaks even after roughly 1,958 recalls if the only comparison
is that 85.75-ms Qwen saving. Incremental summaries and reused features improve
that economics. It does not beat the production BGE route, which already makes
zero query-time Qwen calls.

Against the measured gateway answer time of 23.58 seconds, every local selector
is below 0.004% of the turn. The larger cost is the additional raw evidence:
CAV-2 added about 785 packet tokens, CAV-5 about 770, and spectral about 627
relative to the frozen baseline. More context can also reduce answer quality by
changing emphasis even when baseline evidence remains present.

## Why the 10% target was not reached

The frozen baseline packet contains 35 of 92 exact support quotes. A source-bound
comparison found that CAV-2, CAV-5 and spectral each add **zero** of the 57
missing quotes and remove none of the existing 35. The answer differences among
the scored arms therefore cannot demonstrate improved evidence retrieval. They
measure reader sensitivity, selection emphasis and single-generation variance.

Returning the protected baseline for every question would recover 75.42% from
precomputed CAV-2's 69.65%, an 8.28% relative gain, still below the requested
10%. A post-hoc gate that enables the one historical CAV win would exceed 10%
relative to CAV-2, but that would be selection on the evaluation set and is not
evidence of a deployable classifier.

The result is consistent with recent graph-RAG work. HippoRAG and HippoRAG 2
show that personalized PageRank can help associative memory when the graph
contains useful passage/entity structure. Graph diffusion is also closely
related to spectral filtering. The present graph contains only BGE-neighbor
edges, and its diversity does not reach new checkpoint evidence. Later work on
context-aware traversal specifically warns that static graph walks can drift
through hubs and recommends query-dependent edge weighting and factual
anchors. Confidence-aware reranking separately shows why semantic relevance
does not guarantee generator usefulness.

## Decision and next test

Keep the release default unchanged. Retain `spectral_reinjection` as an opt-in
experimental module because it is fast, sparse and materially different, but
do not enable either injection arm globally.

The next quality experiment must start with questions for which the protected
packet is missing known source support and for which at least one candidate
mechanism actually recovers that support. Use multiple reader generations only
after a provider-free arm passes a novel-support gate. A deployable confidence
gate should preserve the baseline whenever an addition cannot demonstrate
query-specific utility; generator-sampling methods may be tested separately,
but their extra calls must be charged to latency and cost.

## Research basis

- [HippoRAG](https://arxiv.org/abs/2405.14831): knowledge-graph retrieval with personalized PageRank.
- [HippoRAG 2](https://arxiv.org/abs/2502.14802): deeper passage integration while preserving factual retrieval.
- [Diffusion Improves Graph Learning](https://arxiv.org/abs/1911.05485): relationship between localized diffusion and spectral methods.
- [Predict then Propagate](https://arxiv.org/abs/1810.05997): restart-preserving personalized PageRank propagation.
- [Personalized Graph Signal Processing](https://arxiv.org/abs/2302.02113): motivation for retaining personalized/local frequency information.
- [S3Attention](https://arxiv.org/abs/2408.08567), [Perceiver](https://arxiv.org/abs/2103.03206), and [Nyströmformer](https://arxiv.org/abs/2102.03902): alternative low-rank or landmark attention families.
- [CatRAG](https://arxiv.org/abs/2602.01965): query-aware edges and anchors to reduce static-graph drift.
- [CAR](https://arxiv.org/abs/2605.04495): generator-side usefulness and confidence-aware rank correction.

## Evidence

- `examples/cav/engineering_retrieval_concepts_v1.json`: rejected five-narrow-concept screen.
- `examples/cav/engineering_retrieval_five_concepts_v1.json`: admitted five-concept bank.
- `eval_results/cav-five-concept-20261009-r1` and `-r2`: checkpoint-bound fit artifacts.
- `eval_results/cav-reinjection-five-concept-20261009-r1`: prepared packets, timing and two-versus-five comparison.
- `eval_results/cav-five-concept-orthogonal-20261009-r1` and
  `cav-reinjection-five-orthogonal-20261009-r1`: SVD-span correlation control.
- `eval_results/spectral-reinjection-20261009-r1`: sparse graph state, packets, comparison and support analysis.
- `src/memory_condense/associations/spectral_reinjection.py`: sparse restart-preserving filter.
- `tests/test_spectral_reinjection.py`: state, propagation, zero-model and temporal-scope guards.
- `tools/orthogonalize_cav_bundle.py`: reproducible orthonormal span transform.
