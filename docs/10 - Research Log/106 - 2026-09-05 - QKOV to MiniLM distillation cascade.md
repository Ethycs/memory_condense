# QK/OV-to-MiniLM distillation cascade

Date: 2026-09-05  
Status: proxy assay complete; hypothesis failed its safety gate; do not promote

## Decision

Do not replace the Qwen linker with a smaller general language model or a
diffusion language model merely because it has fewer parameters.  On the
three-card Transcript smoke fixture, a matched Qwen3-0.6B prefix did not beat
the Qwen3-8B prefix at short-query latency and lost one of three selections.
The pinned MS MARCO MiniLM cross-encoder instead retained all three selections
at about 11.5 ms mean and 15--16 ms p95 after model warm-up.

The next arm therefore distils only Qwen's **ranking decision** into MiniLM.
It does not claim that a student logit is attention, OV transport, a CAV, or a
replacement for the raw memory payload.

## Intended cascade

1. Query-only lexical/dense routing produces a bounded candidate frontier.
2. MiniLM scores the complete offered frontier once.
3. A frozen, calibration-derived top-one/top-two margin may accept the student
   only when source scope and card coverage are complete and the query does not
   require exhaustive set recovery.
4. Every tie, non-finite score, identity mismatch, truncation, low margin,
   incomplete scope, completeness query, or student exception sends the full
   unchanged frontier to Qwen.
5. If Qwen is unavailable or fails, the selector returns the unpruned original
   frontier and explicitly requires raw fallback.
6. Cards remain routing proxies.  Existing sealed-support verification and raw
   hydration remain the evidence authority.

The student optimizes expected latency only by reducing the fraction of turns
that pay for Qwen.  It cannot improve the latency of a turn that actually
falls back to Qwen.

## Distillation target

The teacher uses independent query/candidate rows through
`QwenMemoryLinker.inspect_coverage`, rendered as:

```text
[Memory]
<candidate>
[Question] <query>
[Readout]
```

For each candidate, the frozen scalar utility is:

```text
max(0, qk_score) + log1p(max(0, ov_transport))
```

Only exact scalar utilities and within-workspace preferences cross the teacher
boundary.  No attention tensors, KV state, transport vectors, answers,
reference evidence, or judge outcomes enter the training artifact.  MiniLM is
trained with a pairwise RankNet objective over candidates from the same teacher
workspace.

## Development firewall

The initial feasibility assay uses only the 200-question development
membership.  It reconstructs that membership from the locked split salt,
question IDs, and public question types, then makes disjoint 140/30/30
train/calibration/test partitions.

The locally available source is the small LongMemEval oracle projection rather
than the 1M-token parent.  To prevent per-question answer-source selection from
becoming a shortcut, every allowed turn is stripped of `has_answer` and other
gold-bearing fields, assigned opaque identities, and pooled across an entire
partition.  Each query receives twelve query-only lexical candidates plus four
salted distractors from that partition-global pool.  No answer source is
injected.

This is deliberately described as a **proxy-card reranker assay**.  It cannot
certify 1M retrieval, Transcript-card behavior, or production accuracy.  A
successful result earns a later shadow run over actual cards from the locked
validation retrieval frontier; it does not earn immediate promotion.

## Promotion gates

The held-out test report must separate:

- zero-shot MiniLM versus the distilled checkpoint;
- student/teacher top-one and pairwise agreement;
- accepted fast-path rate and Qwen-fallback rate;
- student-only p50/p95 and measured end-to-end cascade p50/p95;
- proxy-frontier limitations from full retrieval recall.

The provisional feasibility gates are at least 95% teacher top-one agreement,
at least 97% pairwise agreement, student p50 below 20 ms, and p95 below 50 ms.
Threshold selection uses calibration only.  The test partition remains unseen
until the student and threshold are fixed.

## Implementation

- `src/memory_condense/search/selectors/qkov_distilled_selector.py` contains
  the frozen student identity, policy, native score types, conservative
  shadow/apply selector, and full-frontier teacher/raw fallback behavior.
- `tools/assay_qwen_minilm_distillation.py` builds the sanitized plane, records
  Qwen targets, optionally trains MiniLM, and runs the held-out latency and
  agreement assay.
- `tests/test_qkov_distilled_selector.py` covers confident selection,
  uncertainty, ties, non-finite scores, exceptions, identity mismatch,
  truncation, shadow behavior, and full-frontier fail-open behavior.
- `ContextCardSearchResult.requires_raw_fallback` now also propagates a
  linker's explicit fallback signal through raw hydration.

## Measured result

The first sealed proxy run is a negative result.  The student is fast enough,
but it does not reproduce the Qwen teacher accurately enough to skip Qwen on
any safely calibrated subset.

| Measurement | Result |
|---|---:|
| Development questions | 200 (140 train / 30 calibration / 30 test) |
| Pooled memory turns | 4,206 |
| Qwen teacher work | 400 passes / 3,200 candidate inspections |
| Qwen cold six-layer prefix load | 129.26 s |
| Qwen warm pass latency | 99.53 ms p50 / 118.83 ms p95 |
| Zero-shot MiniLM, calibration top-one | 0/30 (0.0%) |
| Zero-shot MiniLM, calibration pairwise | 1,680/3,600 (46.7%) |
| Zero-shot MiniLM, held-out test top-one | 1/30 (3.3%) |
| Zero-shot MiniLM, held-out test pairwise | 1,801/3,600 (50.0%) |
| Fine-tuned MiniLM, calibration top-one | 13/30 (43.3%) |
| Fine-tuned MiniLM, calibration pairwise | 2,875/3,600 (79.9%) |
| Fine-tuned MiniLM, held-out test top-one | 17/30 (56.7%) |
| Fine-tuned MiniLM, held-out test pairwise | 3,006/3,600 (83.5%) |
| Calibrated student fast path | 0/30 (0.0%) |
| Required Qwen fallback | 30/30 (100.0%) |
| Warm MiniLM scorer latency | 14.27 ms p50 / 19.36 ms p95 |
| Proxy render plus MiniLM | 18.42 ms p50 / 23.16 ms p95 |
| RankNet training | 2 epochs, 16,800 pairs, 136.06 s |

The latency report did not execute live Qwen fallback.  Its observed pipeline
number is only proxy rendering plus MiniLM scoring.  Because calibration sends
every query to Qwen, the executable cascade would retain all Qwen work and add
the student overhead.  It therefore has no speed advantage in this form.

The canonical run artifacts are under `.tmp/qwen-minilm-pilot/`:

- plane: `5e71da688c1937845c271a2972d54a5fd37d3fb948066e8ea9fa7e1d1c054f4b`
- teacher: `741ee0562360c4fd27be698d43b67be9919c8e9759d73f90b14171a9815c3697`
- training: `bd55c53ea7f744ffdb6b5e8819bcc3724d46ac582f370892cb4af541d264cc7d`
- final assay (`assay-v2.json`): `4b55ba35a352f883f06eae80237c15ef97375c7dd78ff1a3f313430a1a15038f`

The projection was repeated independently and produced the same plane digest.

## Interpretation

This rejects the narrow hypothesis that a generic MS MARCO MiniLM
cross-encoder, trained on 140 proxy questions with pairwise QK/OV preferences,
can safely replace this Qwen teacher.  The zero-shot scorer is effectively
uncorrelated with the teacher, and fine-tuning closes only part of that gap.
That is unsurprising: the target is an internal QK-plus-OV routing signal, not
ordinary semantic relevance, and 140 questions provide little coverage of its
decision boundary.

The run does establish that the small-model compute budget is viable.  A
faithful student would make the scorer portion roughly an order of magnitude
cheaper than two Qwen coverage passes.  The unresolved problem is fidelity,
not MiniLM inference speed.

## Validity boundary

This run measures agreement with independent `inspect_coverage` rows over 16
oracle-derived 48-token proxy cards.  It does **not** validate production
`inspect_nested` top-k set selection, its rank-k cutoff margin, LFM2 Transcript
cards, full-corpus candidate recall, a 1M-token ingest, or answer accuracy.
Direct answer/reference fields were absent from the teacher and student
planes, but benchmark truth was used upstream to construct the oracle source.
The training code used only train/calibration partitions, although the sealed
input artifact still contained the test partition; this is code-path
separation rather than a process-level test firewall.

## Next useful arm

Do not spend the next run merely swapping MiniLM for a fashionable Mamba or
diffusion backbone.  The failed component is target fidelity.  A next student
should train on actual production `inspect_nested` traces, optimize the exact
top-k set and cutoff decision, bind rendering/tokenization/max-length into its
identity, and use substantially more gold-free traffic.  A listwise objective
or a smaller Qwen-family early-exit student is the most direct follow-up.  The
same conservative Qwen/raw fallback contract can remain in place.
