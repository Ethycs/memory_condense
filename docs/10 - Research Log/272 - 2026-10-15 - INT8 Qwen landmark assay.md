# INT8 Qwen landmark assay

**Date:** 2026-10-15
**Status:** Promising bounded result; experimental, not release-enabled

## Question

Can the Qwen layer-5 routing state be reduced to eight reusable INT8 K/V
landmarks per summary without losing the useful ranking signal, and can later
questions score those landmarks without re-encoding candidate summaries?

## Implementation

`qwen_int8_landmarks.py` implements a query-independent artifact:

1. project one summary through the existing six-layer Qwen3-8B prefix;
2. take the unrotated layer-5 K/V projections for the summary tokens;
3. partition those tokens into eight contiguous buckets;
4. retain each bucket's mean K and V vector;
5. apply symmetric INT8 quantization with one FP16 scale per KV head; and
6. retain the exact token count represented by each landmark.

The unrotated state is a cross-document semantic index. It is not presented as
an exact causal continuation KV cache. Raw transcript bodies never enter Qwen;
the experiment uses stored summaries and live questions.

For Qwen3-8B's eight KV heads and 128-wide heads, the exact ndarray payload is
16,432 bytes per summary: 16,384 INT8 K/V bytes, 32 scale bytes and 16 token
count bytes. One hundred thousand summaries would retain 1,643,200,000 bytes,
or about 1.53 GiB, before record keys and database framing. Eight candidates
move 131,456 bytes per query.

## Population and execution

The bounded real-data assay reused the 783-summary engineering archive and its
twelve source-bound questions. It compiled the 79 unique summaries in the
frozen BGE top-eight shortlists and reranked eight candidates per question.

The ongoing 500-question benchmark kept both local GPU models resident and left
less than one GiB free. This assay therefore used the full six-layer FP32 Qwen
prefix on CPU with two threads, preserving the benchmark's 401 sealed answers
and resident models. The benchmark continued to 403 answers during the assay.

CPU timings were:

| Operation | Time |
| --- | ---: |
| Warm-file model load | 11.74 s |
| Compile 79 summaries | 227.85 s |
| Compile per summary | 2.884 s |
| Encode 12 query K/V readouts | 15.13 s |
| Query encoding per question | 1.261 s |
| Score all symmetric K/V pairs | 0.038 s |

These CPU values validate execution and artifact reuse. They do not estimate
the intended GPU compile latency. The saved landmark artifact was reused for
the second readout test with zero candidate re-encoding passes.

## Quantization result

The matched FP32-landmark arm isolates INT8 error from eight-landmark pooling
and separately encoded query/candidate error.

| Measurement | Result |
| --- | ---: |
| K relative RMSE | 2.31% |
| V relative RMSE | 1.13% |
| Mean absolute routing-score error | 0.0119 |
| Maximum absolute routing-score error | 0.0443 |
| Mean eight-item rank correlation | 0.9821 |
| Top-one agreement | 10/12 (83.3%) |
| Top-eight set agreement | 12/12 (100%) |

FP32 and INT8 produced identical support metrics at every measured cutoff.
INT8 quantization is therefore not the observed failure source in this sample.

## Readout result

The first readout applied the production linker's length-normalized Q-to-K
statistic to separately encoded, unrotated landmarks. It reduced top-four
support from two quotes to one and recovered no complete fact. That readout is
rejected.

The second readout used the full retained artifact. It mean-pooled the query
readout inside each KV head, compared K-to-K and V-to-V cosine, averaged the
two signals, and voted with the strongest four of eight KV heads. No threshold
or parameter was fitted to the questions.

| Frozen BGE top-eight rerank | Top-4 quotes | Top-4 complete facts | Top-8 quotes | Top-8 complete facts |
| --- | ---: | ---: | ---: | ---: |
| Keep BGE order | 2/92 | 0/42 | 5/92 | 1/42 |
| Raw Q→K landmarks | 1/92 | 0/42 | 5/92 | 1/42 |
| Symmetric INT8 K/V | **4/92** | **1/42** | 5/92 | 1/42 |

Top-eight recall cannot change because all arms preserve the frozen eight-item
candidate set. The relevant result is priority: symmetric K/V moved twice as
many support quotes into the first four slots and completed one fact. The
sample is small, so this is a go signal for a larger test rather than a release
quality claim.

## Decision

Keep the INT8 landmark module and symmetric K/V readout experimental. Do not
replace the production BGE route or current scalar attention cache yet.

The next run should compile all 783 summaries on GPU after the active benchmark
releases its models, measure actual pinned host-to-device transfer, and score
corpus-wide retrieval. A separate archive or blinded question set must then
confirm the priority improvement. The full-corpus assay should compare eight
and sixteen landmarks only if eight landmarks miss source evidence; the current
result gives no reason to pay the larger state cost first.

## Evidence

- `eval_results/qwen-int8-landmark-shortlist-20261015-r1/report.json`
- `eval_results/qwen-int8-landmark-shortlist-20261015-r1/int8-kv-landmarks.npz`
- `eval_results/qwen-int8-landmark-readout-20261015-r1/report.json`
- `src/memory_condense/associations/qwen_int8_landmarks.py`
- `tools/assay_qwen_int8_landmarks.py`
- `tools/assay_qwen_int8_landmark_shortlist.py`
- `tools/assay_qwen_int8_landmark_readout.py`
- `tests/test_qwen_int8_landmarks.py`
