# Predictive V tree probe

**Date:** 2026-10-15  
**Status:** Promising residual-compression diagnostic; not release-enabled

## Question

Can the experimental Qwen V landmarks share storage through a predictive
N-ary hash tree? In particular, can K route to a shared V prototype so that a
memory stores only a short path or residual rather than its full V vector?

## Interpretation

The tested predictive tree is a 4-ary, depth-three vector-quantization tree
with at most 64 leaves per KV head. Two forms were measured:

- **K-routed:** normalized K chooses the leaf and its shared V prototype. The
  path is deterministic from K and costs no additional bits in the V record.
- **V-routed:** V chooses its own prototype. Its six-bit path must be retained.

A continuous ridge map from K to V and direct low-bit V quantization were
included as controls. Every tree, regression and codebook was evaluated with
five-fold cross-validation grouped by summary. No summary supplied vectors to
its own predictor. The probe reused 79 summaries, eight landmarks, eight KV
heads and width 128 from the existing development artifact. It made no Qwen,
provider or raw-section calls.

## Prototype-only result

The N-tree does not break V down sufficiently by itself.

| Predictor | Mean V cosine | Proxy top-1 agreement | Proxy top-8 overlap |
| --- | ---: | ---: | ---: |
| K-routed 64-leaf tree | 0.630 | 43.0% | 54.7% |
| V-routed 64-leaf tree | 0.676 | 40.5% | 58.5% |
| Continuous K-to-V ridge | **0.708** | **62.0%** | **65.5%** |

Six path bits cannot represent a high-dimensional V direction. Increasing the
branching factor at the same path-bit budget is unlikely to alter that basic
rate limit. The tree must predict a residual rather than replace V.

## Fixed-rate residual result

| Representation | Bits/dimension | Mean V cosine | Proxy score MAE | Proxy top-1 | Proxy top-8 |
| --- | ---: | ---: | ---: | ---: | ---: |
| Direct 3-bit V | 3 | 0.9485 | 0.01175 | 82.3% | 87.5% |
| K-tree + 3-bit residual | 3 | 0.9692 | 0.00464 | 82.3% | 92.7% |
| Ridge + 3-bit residual | 3 | **0.9748** | **0.00375** | **89.9%** | **93.2%** |
| V-tree + 3-bit residual | 3 plus path | 0.9749 | 0.00406 | 86.1% | 92.6% |
| Direct 4-bit V | 4 | 0.9898 | 0.00237 | 92.4% | 95.7% |
| K-tree + 4-bit residual | 4 | 0.9940 | 0.00117 | **94.9%** | 96.2% |
| Ridge + 4-bit residual | 4 | 0.9952 | **0.00098** | 91.1% | 96.2% |
| V-tree + 4-bit residual | 4 plus path | **0.9952** | 0.00103 | 91.1% | **97.0%** |

Four-bit residuals require an ideal 64 data bytes plus a two-byte FP16 scale
per 128-wide V vector, versus 128 data bytes for current INT8 V. Across the 64
V vectors in one eight-landmark summary, that is 4,224 bytes rather than 8,192
bytes before codebook amortization. The tree improves fidelity at that rate;
it does not reduce the fixed-width payload beyond ordinary four-bit coding.

## Adaptive-rate result

Each vector selected the lowest of two, three or four residual bits that met a
cosine target, otherwise falling back to eight bits.

| Predictor | Target 0.98 | Target 0.99 | Target 0.995 |
| --- | ---: | ---: | ---: |
| Direct quantization | 4.161 bits/dim | 5.594 | 7.801 |
| K-routed tree | 3.723 | 4.384 | 5.978 |
| Continuous K-to-V ridge | **3.557** | 4.156 | **5.407** |
| V-routed tree | 3.574 | **4.106** | 5.530 |

The V-routed figures exclude its six-bit path, equal to 0.0469 bits per
dimension. All adaptive figures also exclude a small bit-width selector and
scale metadata. After charging the V-tree path, the continuous ridge predictor
is approximately tied with or better than the tree at the strict targets and
requires no per-vector path.

At the 0.99 target, the K-routed tree reduces the residual data rate by about
21.6% relative to adaptive direct quantization. Relative to the current fixed
eight-bit V, its 4.384-bit data rate is about 45% smaller before scale and
selector metadata. A four-bit fixed residual is smaller still, but does not
guarantee that every vector meets the target.

## Decision

Do not use a prototype-only V hash tree. Keep the idea as a predictor for a
multi-landmark detail record.

The next full-corpus experiment should use:

1. a per-head continuous K-to-V predictor as the primary control;
2. a K-routed tree as the interpretable/discrete comparator;
3. residual vector or product quantization over the prediction error;
4. variable-rate two/three/four/eight-bit residuals selected by a fixed
   reconstruction threshold;
5. exact question readouts and source-support scoring, because the present
   summary-to-summary rankings are only a geometry proxy; and
6. codebook bytes, path bytes, selectors, scales and decode latency in every
   storage claim.

The tree is relevant only if multi-landmark late interaction improves support.
For the current centroid-only scorer, the 2,080-byte per-summary centroid code
from the preceding probe is both simpler and substantially smaller.

With a larger publication population, test residual vector quantization rather
than making one tree extremely wide. Residual VQ successively quantizes the
remaining error and is the established form of this idea for high-dimensional
ANN compression: [Approximate Nearest Neighbor Search by Residual Vector
Quantization](https://pmc.ncbi.nlm.nih.gov/articles/PMC3231071/). Tree-structured
VQ reduces search complexity but does not remove the rate-distortion limit:
[Design and performance of tree-structured vector
quantizers](https://doi.org/10.1016/0306-4573(94)90012-4).

