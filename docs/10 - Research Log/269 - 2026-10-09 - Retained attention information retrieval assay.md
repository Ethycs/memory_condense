# Retained attention information retrieval assay

**Date:** 2026-10-09  
**Status:** Positive for dialogue-wide synthesis; independent explicit-fact holdout rejects general promotion

## Question corrected

The earlier architecture discarded Qwen token activations after reducing them
to scalar hierarchy-link scores. The stated reasons mixed three separate
claims:

1. attention state would be too large to retain;
2. token-by-token attention maps are awkward retrieval objects because their
   axes refer to positions inside different texts; and
3. the discarded state had not shown retrieval value beyond BGE.

The first claim does not fit the system being evaluated. This archive contains
783 stored summary sections, not a million independently indexed objects. The
second claim remains true for raw `[heads, tokens, tokens]` maps, but does not
require discarding every attention-derived representation. The third claim was
empirical and needed a retrieval assay.

## Representation tested

The assay captures the layer-4 input to Qwen's attention output projection:
the concatenated per-head `A @ V` result before `W_O`. It mean-pools valid
tokens into one 4,096-value vector per stored summary or question. This keeps
attention-conditioned content in a shared coordinate system while releasing
token activations and position-specific attention maps after every batch.

Qwen saw only the 783 stored summaries and twelve live questions. It never saw
raw transcript bodies. Exact raw sections were used only after ranking to
measure authenticated support coverage. No provider or reader calls were
made.

The population is the sealed 775,741-token engineering dialogue used by the
native-experiment comparison: twelve questions, 42 source-bound facts and 92
exact support quotes. Its protected production packets contain 35 quotes and
eleven complete facts.

## Does the extra information retrieve anything?

Yes. Direct cosine retrieval in the full attention-value space reaches evidence
that the production packet and an isolated BGE search miss.

| Added candidates per question | BGE: new quotes / facts | Full `A @ V`: new quotes / facts | BGE seed plus stored `A @ V` neighbors: new quotes / facts |
| ---: | ---: | ---: | ---: |
| 8 | 1 / 0 | 0 / 0 | 0 / 0 |
| 16 | 1 / 0 | **4 / 2** | 0 / 0 |
| 32 | 2 / 1 | **9 / 3** | **4 / 1** |
| 64 | 3 / 1 | **10 / 4** | **6 / 2** |

The direct attention and BGE top lists are materially different: their mean
Jaccard overlap is 0.092 at eight candidates, 0.146 at sixteen, 0.168 at 32 and
0.173 at 64. Equal reciprocal-rank fusion wastes that complementarity at small
budgets, recovering only one new quote through 32 candidates. The signal needs
a protected-baseline union or a learned/gated candidate policy rather than a
naive average.

The stored-only neighbor arm is weaker, but its gains matter: BGE can seed a
graph built from persisted attention features and recover four new quotes at
32 candidates without running Qwen for the question. The retrieval value is
there even when the live route does not pay a query-time Qwen pass.

## Compact finite basis

An unsupervised SVD was fitted on stored summary features only. Questions and
support labels did not participate. Projecting documents and questions into an
uncentered 16-dimensional basis improved the small-budget result:

| Added candidates | Isolated code hits | Production union | Gain over production |
| ---: | ---: | ---: | ---: |
| 8 | 5 quotes / 2 facts | 40 quotes / 13 facts | **+5 quotes / +2 facts** |
| 16 | 14 quotes / 5 facts | 43 quotes / 14 facts | **+8 quotes / +3 facts** |
| 32 | 15 quotes / 5 facts | 43 quotes / 14 facts | **+8 quotes / +3 facts** |
| 64 | 26 quotes / 7 facts | 48 quotes / 15 facts | **+13 quotes / +4 facts** |

At eight candidates, gains occur on four different questions: chronological
inventory, architecture evolution, latency diagnosis and beta readiness. At
sixteen candidates they also reach the unresolved-problems synthesis. This is
not one repeated support span or one lucky question cluster.

The 16-dimensional result likely benefits from denoising: it explains 68.44%
of uncentered feature energy and beats the full 4,096-dimensional vector at the
smallest budgets. Rank selection was inspected on this development archive,
however, so that advantage is not yet an unbiased estimate.

## Storage and latency

| State or operation | Measured result |
| --- | ---: |
| Native memory directory | 112,279,132 bytes |
| Full FP16 `A @ V` document vectors | 6,414,464 bytes |
| FP16 all-document similarity matrix, if retained | 1,226,178 bytes |
| Existing scalar attention cache | 30,424 bytes |
| Rank-16 FP16 document codes | 25,056 bytes |
| Shared rank-16 FP16 basis | 131,072 bytes |
| Rank-16 total | **156,128 bytes** |
| Cold Qwen preparation | 98.5945 s |
| Encode all 783 summaries after preparation | 8.1652 s |
| Encode twelve questions in two batches | 0.1374 s |
| BGE CPU encoding for the same twelve questions | 1.2555 s |

The mean Qwen query figure of 11.45 ms is a batched amortized measurement, not
a single interactive-query latency claim. Resident single-query timing remains
a release gate. FP16 persistence reproduced every full-vector retrieval count
and top-eight ranking exactly.

At this archive size, neither the 6.1 MiB full representation nor the 152.5 KiB
rank-16 representation creates a credible storage problem. The cold model load
is the material cost, and publication should create new codes incrementally
while the already-resident hierarchy compiler is available.

## Independent 100-question holdout

Rank 16 and an eight-candidate cap were fixed before opening a different
1.115M-token archive. This archive contains 5,483 stored summaries and 100 new
questions grounded in 191 exact user quotes. Its production packets already
cover 187/191 quotes and completely support 97/100 questions.

| Arm | Exact supports | Complete questions | New over production |
| --- | ---: | ---: | ---: |
| Production packet | 187 / 191 | 97 / 100 | Reference |
| Rank-16 code, top 8 alone | 5 / 191 | 2 / 100 | — |
| Production union rank-16 top 8 | 187 / 191 | 97 / 100 | **0 quotes / 0 questions** |
| Rank-16 code, top 16 alone | 7 / 191 | 4 / 100 | — |
| Production union rank-16 top 16 | 187 / 191 | 97 / 100 | **0 quotes / 0 questions** |

This does not replicate the development gain as a general fact retriever. It
does clarify the operating region. The development questions require
chronology, revision tracking, unresolved-state synthesis and links across
distant engineering episodes. The holdout asks for explicit user facts, where
the existing hierarchy and BGE route are already near ceiling. Attention codes
remain a targeted candidate for cross-episode synthesis, not a replacement for
the direct route.

Cold preparation took 17.59 seconds, encoding 5,483 summaries took 74.87
seconds, and encoding 100 questions took 0.918 seconds. Rank-16 FP16 document
codes for this archive would occupy 175,456 bytes; including the shared basis,
the retained state would be 306,528 bytes.

## Token-level observable

The retained code has an exact token attribution. For source token `j`, define

\[
c_j=\frac{1}{T}\sum_i A_{ij}V_j.
\]

The source-token vectors sum to the mean-pooled `A @ V` document feature.
Applying the same rank-16 basis and taking each token's term in the normalized
document-question dot product makes the token terms sum to the retrieval
cosine.

One held-out exact-support hit provides a numerical check. For the question
“Which Mass am I planning to attend at St. Mary’s this Sunday?”, the code chose
the summary “User plans to attend the 9:30 am mass at St. Mary's this Sunday
and asks for that day's Gospel reading.” Its rank-16 cosine was 0.873967826;
the token terms summed to 0.873967946, an absolute reconstruction error of
1.19e-7.

The largest positive terms were `User` (0.2242), `plans` (0.1180), `attend`
(0.0989), `to` (0.0656), `the` (0.0516), `9` (0.0432), `asks` (0.0426),
`mass` (0.0215) and `am` (0.0203). This trace connects tokenizer positions,
attention weights, values, the finite basis and the final retrieval score. It
also exposes a concrete tuning opportunity: generic summary framing such as
`User` carries substantial score mass and may be better stripped or modeled in
a separate channel.

The matrices do not need to be stored to preserve auditability. The system can
retain the compact document code and reconstruct token contributions on demand
by rerunning Qwen for the few selected summaries. A deeper trace can retain
destination token `i` and head `h` in `A[h,i,j]V[h,j]` to show which head and
destination token transported each source token.

## Qwen-token TF-IDF correction

The large `User` contribution motivated a sparse rarity control using the same
pinned Qwen tokenizer IDs. TF-IDF was compiled independently over each corpus's
stored summaries and compared with rank-16 attention and two fixed
reciprocal-rank fusions.

| Archive and top-eight arm | Isolated exact support | New support over production | Newly complete units |
| --- | ---: | ---: | ---: |
| Engineering synthesis: TF-IDF | 2 / 92 | 0 | 0 / 42 facts |
| Engineering synthesis: rank-16 attention | 5 / 92 | **5** | **2 / 42 facts** |
| Engineering synthesis: equal fusion | 5 / 92 | 3 | 1 / 42 facts |
| Explicit-fact holdout: TF-IDF | **109 / 191** | **1** | **1 / 100 questions** |
| Explicit-fact holdout: rank-16 attention | 5 / 191 | 0 | 0 / 100 questions |
| Explicit-fact holdout: equal fusion | 80 / 191 | 0 | 0 / 100 questions |

At sixteen candidates, TF-IDF added three engineering support quotes and one
complete fact, but rank-16 attention added eight and three. On the held-out
archive, TF-IDF's one new quote completed question Q022: “What themes did I
want upcoming art exhibition or event recommendations to focus on?” It
recovered the missed exact statement about “artwork created by local kids or
underrepresented communities.”

TF-IDF compiled in 0.60 seconds for 783 summaries and 1.70 seconds for 5,483
summaries. The larger sparse matrix contained 209,953 nonzero document cells.
It needs no Qwen forward pass; only the already pinned tokenizer vocabulary.

The useful policy is conditional selection, not score averaging. For explicit
entities, numbers, names and literal requirements, TF-IDF supplies a cheap
rarity-sensitive candidate branch. For chronology, revision and cross-episode
synthesis, the attention code remains stronger. Equal fusion weakened the
development attention result, and weighting attention twice as strongly still
lost one novel quote and one complete fact at the eight-candidate cap.

IDF-weighting each source-token transport term before attention pooling remains
a distinct experiment:

\[
z_{\mathrm{idf}}=\frac{1}{T}\sum_i\sum_j
\operatorname{idf}(t_j)A_{ij}V_j.
\]

It would directly suppress boilerplate inside the dense code, but requires
recomputing and validating all stored codes. The sparse TF-IDF branch already
recovers the held-out missing fact, so the transport-weighted variant should be
tested only as a synthesis-code refinement, with the unweighted code preserved
as its control.

## Decision

The blanket discard rationale is rejected. Raw token-by-token attention maps
should remain transient because they are position-specific and unnecessary for
the measured win. A compact pooled attention-value code should be retained as
an experimental retrieval feature before the activations are released.

This is a specialized candidate producer, not a default packet expansion. The
independent holdout rejects always-on use for explicit-fact retrieval. Further
testing should gate it to dialogue-wide chronology, revision and synthesis
questions, keep the production packet protected, and measure whether a
boilerplate-corrected code repeats the development support gain on a second
engineering or research archive. Single-query resident latency, temporal-scope
guards, an incremental-basis policy and reader-level answer confirmation also
remain required.

## Evidence

- `eval_results/attention-feature-retrieval-20261009-r1/report.json`: full
  representation, BGE control, stored-neighbor arm, timings and storage.
- `eval_results/attention-feature-retrieval-20261009-r1/rankings.json`: frozen
  per-question rankings.
- `eval_results/attention-feature-retrieval-20261009-r1/attention-values-fp16.npy`:
  persisted summary features.
- `eval_results/attention-feature-basis-20261009-r2/report.json`: basis ranks,
  complete storage accounting and per-question gain audit.
- `eval_results/attention-feature-holdout-20261009-r1/report.json`: predeclared
  rank-16, cap-eight result on 100 independently authored questions.
- `eval_results/attention-token-observable-20261009-r1/report.json`: exact
  token-to-retrieval-score reconstruction for one support-bearing candidate.
- `eval_results/tfidf-attention-routing-20261009-r2/report.json`: Qwen-token
  TF-IDF and fixed-fusion screen on development and holdout archives.
- `tools/assay_attention_feature_retrieval.py`: provider-free retrieval assay.
- `tools/analyze_attention_feature_basis.py`: unsupervised finite-basis assay.
- `tools/validate_attention_feature_holdout.py`: held-out support gate.
- `tools/trace_attention_token_observable.py`: on-demand token attribution.
- `tools/assay_tfidf_attention_routing.py`: sparse rarity-control assay.
- `src/memory_condense/modeling/qwen_prefix.py`: bounded `A @ V` capture.
