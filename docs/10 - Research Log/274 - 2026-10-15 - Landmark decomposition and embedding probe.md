# Landmark decomposition and embedding probe

**Date:** 2026-10-15  
**Status:** Artifact-only development diagnostic; no release change

## Question

Can the experimental eight-landmark Qwen K/V records be decomposed or indexed
as embeddings, and would either operation improve the memory system?

## Existing behavior

The successful experimental symmetric readout does not currently compare the
eight landmarks individually. It normalizes them, computes a token-count
weighted centroid inside each KV head, and then compares query and candidate
K/K and V/V centroids. The score is the mean of the strongest four of eight
head scores.

Therefore one K centroid and one V centroid per head are sufficient to
reproduce the present readout before a second quantization step. Retaining all
eight landmarks only becomes useful if a later readout uses their internal
structure, such as late interaction or hierarchical residual refinement.

## Artifact-only probe

The probe reused the 79-summary development artifact at
`eval_results/qwen-int8-landmark-shortlist-20261015-r1/int8-kv-landmarks.npz`.
It did not run Qwen, access raw sections, alter the active benchmark, or tune
against answer labels.

### INT8 centroid embedding

Each stored record was reduced to the exact per-head normalized centroid used
by the current symmetric scorer, then quantized again to INT8 with one scale
per head.

| Measurement | Result |
| --- | ---: |
| Existing eight-landmark record | 16,432 bytes/summary |
| One K and V centroid per head | 2,080 bytes/summary |
| Size reduction | 7.90x |
| K centroid relative RMSE | 1.99% |
| V centroid relative RMSE | 0.98% |
| Pairwise proxy-score MAE | 0.000247 |
| Pairwise proxy top-one agreement | 96.2% |
| Mean proxy top-eight set overlap | 99.2% |

The ranking comparison used stored summaries as pseudo-queries because query
K/V tensors were not retained in the prior assay. It demonstrates geometric
parity, not question-level retrieval parity. The post-benchmark run must repeat
the comparison with real question readouts and exact-support metrics.

### Per-head low-rank decomposition

An uncentered SVD was fitted separately per KV head to the 632 landmark vectors
(79 summaries by eight landmarks). This is a development spectrum only; any
durable basis must be fitted without holdout questions or labels.

| Width | K energy retained | V energy retained |
| ---: | ---: | ---: |
| 16 / 128 | 96.82% | 66.32% |
| 32 / 128 | 98.37% | 79.19% |
| 64 / 128 | 99.59% | 91.70% |

Keys are strongly low-rank in this sample. Values are substantially less
compressible and should not inherit the same rank merely for symmetry.

### Decomposition over the eight landmark positions

An orthonormal DCT was applied over the contiguous landmark axis after
normalizing each vector.

| Prefix coefficients retained | K energy | V energy |
| ---: | ---: | ---: |
| 1 / 8 | 92.86% | 40.95% |
| 2 / 8 | 94.16% | 52.16% |
| 4 / 8 | 96.32% | 69.96% |
| 8 / 8 | 100% | 100% |

The K sequence is dominated by a coarse component. V carries much more local
detail. This independently argues for thin/coarse keys and fuller values.

## Recommended architecture

Use two tiers rather than choosing between one embedding and all landmarks:

1. **Coarse Qwen-native index.** Store one INT8 K and V centroid per head.
   Search it as a block-aware embedding, union candidates across heads, and
   apply the existing strongest-head scorer to the shortlist. A standard
   single-vector dot product cannot exactly reproduce the top-four-of-eight
   head vote, so the ANN stage remains a coarse generator.
2. **Optional detail record.** Retain landmark residuals only for candidates
   that need refinement. Try a low-rank K basis first. Preserve more V width
   and landmark detail until support tests show it is disposable.
3. **Late-interaction readout.** Compare query landmarks with candidate
   landmarks before pooling, analogous to multi-vector late interaction. This
   is the quality arm most likely to help multi-topic summaries whose relevant
   feature is diluted by the centroid.
4. **Exact hydration remains unchanged.** Every code stays bound to summary,
   section occurrence, conversation, checkpoint, layer and source hash. The
   selected raw section is hydrated only after routing.

Do not translate the codes into BGE space first. BGE is already the strong
production candidate generator, and an adapter would add approximation without
testing whether Qwen-native geometry contributes novel support. Test a native
Qwen landmark index and a BGE-top-64 reranker first.

## Post-benchmark arms to add

Alongside the predeclared eight-landmark arms, measure:

- real-query parity of the 2,080-byte centroid record;
- per-head K ranks 16, 32 and 64 while retaining full-width V;
- centroid-only retrieval versus multi-landmark MaxSim/softmax retrieval;
- coarse centroid ANN followed by exact strongest-head reranking;
- optional residual loading for low-margin or high-entropy queries; and
- storage, page faults, transfer time, total recall latency, exact supports,
  complete questions and novel support over the protected packet.

Promote the centroid representation only if it loses no complete question and
at most one exact support on the 100-question holdout. Promote landmark detail
only if it recovers novel support or enables at least 20% packet compression
without reader regression.

## Relation to published methods

- [ColBERT](https://arxiv.org/abs/2004.12832) shows why independently encoded
  multi-vector records plus cheap late interaction can support direct indexed
  retrieval.
- [Eigen Attention](https://arxiv.org/abs/2408.05646) provides precedent for
  placing attention K/V state in a learned low-rank space.
- [Thin Keys, Full Values](https://arxiv.org/abs/2603.04427) reports that keys
  can be much more compressible than values, matching this development probe's
  asymmetry.
- [Product Quantization for Nearest Neighbor Search](https://doi.org/10.1109/TPAMI.2010.57)
  supplies the standard route from compact subspace codes to scalable ANN.
- [Landmark Attention](https://arxiv.org/abs/2305.16300) demonstrates
  attention-native block routing, but its trained landmark tokens are not the
  same mechanism as this project's post-hoc layer-5 summary centroids.

