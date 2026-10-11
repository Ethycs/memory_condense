# Precomputed CAV reinjection compute assay

**Date:** 2026-10-09  
**Status:** Complete; compute claim supported, release promotion rejected by matched answer quality

## Question corrected

The October 8 `cav_associations` arm tested CAVs as an accuracy-oriented
query-time Qwen reranker. That is not the original reinjection hypothesis. The
original proposal used a small concept bank as a low-rank replacement for
memory-wide attention:

\[
E=\operatorname{softmax}(\widehat C_0\widehat X^T/\tau_e),\quad
C_1=EX,\quad
R=\operatorname{softmax}(\widehat X\widehat C_1^T/\tau_r).
\]

For `N` stored summaries and `K` concepts, the persistent topology is `[K,N]`
plus `[N,K]`, rather than `[N,N]`. The previous native adapter still ran one
Qwen prefix pass for every question and therefore could not establish the
compute claim.

## Implemented arm

`cav_reinjection` now performs the Qwen-dependent work at publication:

1. Mean-pool the existing layer-4 Qwen residual for each exact stored summary.
2. Reuse features by checkpoint, layer and exact summary text.
3. Build `E` and `R` with the checkpoint-bound CAV bank.
4. Discard the 4,096-wide summary residuals from the live index.
5. At recall, reuse the production BGE query and direct-route scores as seeds.
6. Propagate the seeds through `E` and `R`, fuse the result geometrically with
   the existing BGE shortlist scores, and hydrate exact raw spans normally.

The BGE shortlist constraint was added after an intentionally stopped first
readout. Global two-concept propagation repeatedly chose generic tool and
progress turns. It was computationally cheap but had discarded too much query
identity. No reader calls were made on that version. The corrected arm uses CAV
reinjection as a low-rank reranker, rather than a corpus-wide replacement for
semantic addressing.

No raw transcript text enters Qwen. Query recall runs no Qwen pass, retains no
transformer token state and constructs no evidence-pair matrix. Historical
as-of queries whose eligible source set differs from the compiled publication
fail closed instead of receiving a concept state influenced by future rows.

## Reused population

The assay reuses the already-ingested 775,741-token, 481-message engineering
archive from Research Log 266, its BGE vectors, twelve questions, source-bound
checkpoints, baseline packets and baseline Sol answers. No history was
reingested. Exact comparison confirmed that current reader-visible baseline
text and raw references match the sealed prior inputs despite newer receipt
metadata.

The current bank contains only two broad development concepts:
`context_dependency` and `binding_constraint`. Its concept classification
score is not evidence-utility accuracy.

## Compute result

| Measurement | Result |
| --- | ---: |
| Atomic summary rows | 783 |
| CAV concepts | 2 |
| Hidden width used during publication | 4,096 |
| Cold Qwen feature batches | 98 |
| Cold state preparation | 167.9347 s |
| Warm cached feature batches | 0 |
| Warm state preparation | 2.8550 s |
| Retained `E + R` state | 12,528 bytes / 3,132 FP32 cells |
| Mean query reinjection readout | 0.3873 ms |
| Mean query interaction cells | 0.007829% of a 783-by-783 dense map |
| Query Qwen passes | 0 |
| Paired baseline full recall | 1.0198 s |
| CAV-reinjection full recall | 1.0696 s |

The 49.8 ms full-recall difference is raw-span selection and packing plus run
noise, not latent compute. The reinjection readout itself is below one
millisecond. This demonstrates the intended asymptotic and measured local
compute reduction relative to the query-time Qwen CAV arm or a dense latent
read. It is not a latency win over production BGE-only recall, which already
runs no query-time Qwen attention. It does not show that the new passages
improve answers.

## Retrieval result

The corrected arm adds 2.67 atomic sections on average. Baseline evidence is
preserved exactly. Frozen checkpoint support remains **11 complete checkpoint
sets and 35 of 92 exact support quotes**, identical to the baseline and every
October 8 experimental arm. The additions include some relevant implementation
and failure evidence, but also tool receipts and progress reports. That makes
answer-time distraction the remaining gate.

Eleven of twelve treatment prompts differ from baseline. The completed bounded
gateway phase made eleven Sol answer calls and twelve blinded Sol judge calls.
Actual usage was 362,868 prompt tokens and 6,863 completion tokens: 330,702
answer-prompt tokens plus 32,166 judge-prompt tokens. Mean gateway time was
23.58 seconds per treatment answer and 18.18 seconds per paired grade. These
times measure the remote reader and judge, not local recall.

## Matched answer result

| Arm | Audited points | Audited macro | Wins / ties / losses vs baseline |
| --- | ---: | ---: | ---: |
| Frozen default memory route | 60 / 84 | 75.42% | 0 / 12 / 0 |
| Precomputed CAV reinjection | 54 / 84 | 69.65% | 0 / 8 / 4 |
| Prior query-time CAV association | 57 / 84 | 70.14% | 1 / 8 / 3 |

The raw second judge scored the frozen baseline at 55/84 even though its answer
identities, source-bound rubrics and reader-visible inputs were reused from the
sealed October 8 assay, where they had already been semantically audited at
60/84. The raw judge also gave zero on Q07/F12 despite the treatment answer
explicitly stating the 6.0% mixed-granularity collapse, short-text cosine
domination and stratified correction. Q12 bound both arms to the same response
identity but received scores inconsistent with the frozen audit. The immutable
raw grades remain in place. `scoring-adjudication.json` records eight explicit
checkpoint corrections, their rationales, source hashes and final totals.

The treatment tied all three focused lookups. Its four losses were the broad
chronological inventory Q02 (-3 points), assistant-mistake synthesis Q03 (-1),
accuracy-change diagnosis Q07 (-1), and solved-then-failed inventory Q09 (-1).
Baseline support was never removed, so the losses are reader distraction or
selection emphasis rather than evidence deletion. The two-concept state often
adds relevant material, but it also adds generic implementation and progress
evidence that competes with the question's requested chronology or failure
boundary.

The important comparison for the original compute hypothesis is the prior
query-time CAV arm. Precomputation reduced query Qwen passes from one to zero
and lowered mean recall from 1.155 seconds to 1.070 seconds, while audited macro
quality changed from 70.14% to 69.65%. This is close on one small assay, but it
is not statistical equivalence: the new arm scored three fewer raw checkpoint
points and had one additional question-level loss.

## Decision

The compute hypothesis is supported locally: precomputed CAV reinjection can
replace the query-time Qwen CAV pass with a sub-millisecond low-rank readout and
a 12.2 KiB live state while retaining roughly the prior CAV arm's answer
quality. It remains opt-in because both CAV variants trail the default BGE
memory route, cold backfill is 168 seconds, and the current two-concept bank is
too broad for reliable evidence utility.

The next experiment should train concepts for evidence utility, currentness and
question type, then enforce a confidence gate that returns no additions when
the CAV distribution is diffuse. Promotion still requires a larger multi-session
assay and an incremental-ingestion measurement showing that new feature
production stays behind interactive replies.

## Evidence

- `eval_results/cav-reinjection-quality-20261009-r1/state.json`: cold build.
- `eval_results/cav-reinjection-quality-20261009-r5`: corrected frozen packets,
  timing, responses, immutable grades, raw report and scoring adjudication.
- `eval_results/cav-reinjection-feature-cache-v1`: content-addressed Qwen
  summary features used by warm publication.
- `tools/characterize_cav_reinjection.py`: incremental prepare, generation,
  grading and report lifecycle.
- `tools/adjudicate_cav_reinjection_scoring.py`: reproducible transparent
  correction layer for the frozen paired grades.
- `tests/test_cav_reinjection.py` and `tests/test_native_experiments.py`:
  rectangular-state, namespace, temporal-scope, zero-Qwen and exact-hydration
  guards.
