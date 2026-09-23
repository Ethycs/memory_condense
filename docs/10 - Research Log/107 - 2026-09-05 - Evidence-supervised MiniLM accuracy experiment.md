# Evidence-supervised MiniLM accuracy experiment

Date: 2026-09-05  
Status: complete; advance gate failed; not a policy claim

## Why the target changed

The first QK/OV-to-MiniLM pilot established that MiniLM is fast but cannot
safely copy the independent Qwen coverage ranking from 140 proxy questions.
More importantly, copying that ranking is not the retrieval objective. On the
prior 30-question proxy test, the candidate frontier contained target evidence
for 30/30 questions, while Qwen's independent utility ranking contained it at
6/30 at rank one, 19/30 by rank four, and 25/30 by rank eight.

The next experiment therefore optimizes and scores **evidence containment**.
Qwen agreement is diagnostic only. No new Qwen calls are required for this
arm.

## Validity boundary

There is no useful-scale production ContextCard corpus in the workspace. The
only accepted LFM2 Transcript artifact contains three synthetic cards; its
60-row timing artifacts repeat those same three cases. The available
LongMemEval oracle projection contains only answer sessions upstream, and all
200 development questions have already participated in analysis.

Accordingly this is a five-fold, question-level, out-of-fold **development
experiment**. It is not a fresh holdout, not a 1M-token run, and cannot promote
the retrieval policy. The locked validation100 and confirmation200 populations
remain closed.

## Frontier ceiling assay

The experiment reranks a partition-global query-only BM25 frontier. A
pre-training ceiling assay found:

| Frontier | Any-evidence ceiling | Exact annotated-turn ceiling | Mean required-source ceiling | All required sources present |
|---:|---:|---:|---:|---:|
| 16 | 186/200 | 162/200 | 85.18% | 154/200 |
| 32 | 191/200 | 169/200 | 89.70% | 166/200 |
| 48 | 193/200 | 173/200 | 92.97% | 177/200 |
| 64 | 193/200 | 177/200 | 93.78% | 180/200 |
| 96 | 196/200 | 179/200 | 96.03% | 186/200 |
| 128 | 198/200 | 179/200 | 97.65% | 192/200 |

A 32-item frontier cannot satisfy a 95% multi-source objective. The frozen
accuracy arm therefore uses lexical 96 and reranks it to eight. Latency for
the larger frontier is measured rather than assumed.

## Frozen experiment

- Population: all 200 analysis-used development questions.
- Evaluation: five deterministic stratified folds of 40 questions. Each fold
  starts from the same pinned base checkpoint and optimizes only the other 160
  query labels.
- Candidate text: 48-token deterministic raw-turn proxy.
- Query text: 48 tokens; pair maximum: 128 tokens.
- Relevance grade 2: an exact `has_answer` turn.
- Relevance grade 1: another turn from a required answer session.
- Relevance grade 0: a turn from another question's oracle sessions.
- Model: pinned `cross-encoder/ms-marco-MiniLM-L6-v2`.
- Optimizer: AdamW, learning rate `2e-5`, weight decay `0.01`, three epochs,
  ten-percent warm-up, gradient clipping at 1.0, effective four-question
  accumulation, deterministic fold seeds.

For a positive set `P`, the listwise boundary term is:

```text
H(P, k) = softplus(0.2 + kth_negative_score(k)
                   - 0.25 * logmeanexp(positive_scores / 0.25))
```

The total loss combines any-evidence-at-eight, equal-weight required-source
coverage, exact-turn-at-eight when an exact annotation is present, and a
smaller any-evidence-at-one term. The full 96-score list participates in each
boundary; training is not pair-sampled.

## Metrics and advance gate

The report separates lexical order, zero-shot MiniLM, and out-of-fold trained
MiniLM. It records any-evidence hit@1/4/8, MRR, exact-turn containment,
mean/all required-source recall at eight, per-fold and per-question-type
results, lexical rescues/regressions, and warm latency.

The preregistered advance gate is:

- at least 190/200 out-of-fold any-evidence hit@8;
- mean required-source recall at least 95%;
- exact annotated-turn hit@8 at least 95% among the 179 questions whose
  annotated turn is reachable in the lexical-96 frontier (at least 171/179);
- no more than two regressions against lexical top eight;
- no more than two exact-turn regressions against lexical top eight;
- at least 37/40 hit@8 in every fold; and
- warm p95 below 50 ms.

Passing earns a separate sealed shadow on validation100. It does not itself
change production. Failure retires this generic MiniLM objective before any
additional Qwen or provider work.

## Results

The five-fold CUDA run completed and published the canonical artifact at
`.tmp/minilm-evidence-accuracy-oof-v1-final.json` with SHA-256
`f82a6845f0f7d4b2752ca598dd4828e4589cd9a54d28cd872f32d5de08a13f80`.
The sidecar and an independent file hash agree, and the embedded assay-source
SHA-256 `55f7fbfa…70c3f` matches the current implementation. The final command
took roughly seven minutes wall clock; fold training accounted for 386.14
seconds and fresh model loads for 10.66 seconds.

An earlier artifact named `minilm-evidence-accuracy-oof-v1.json` was produced
while the final source-identity receipt was being added and is superseded. The
provenance-correct rerun reproduced the fold assignment, frontiers, gold
coordinates, all base and trained score hashes, every trained fold state,
aggregate metrics, comparisons, and gate checks exactly. Only runtime timing
and receipt bytes differ.

| Metric | Lexical | Base MiniLM | OOF trained MiniLM |
|---|---:|---:|---:|
| Any required session @1 | 147/200 | 141/200 | 149/200 |
| Any required session @4 | 170/200 | 169/200 | 177/200 |
| Any required session @8 | 178/200 | 177/200 | 181/200 |
| Exact annotated turn @1 | 82/200 | 48/200 | 47/200 |
| Exact annotated turn @4 | 140/200 | 92/200 | 110/200 |
| Exact annotated turn @8 | 155/200 | 114/200 | 127/200 |
| Exact @8, conditional on 179 reachable annotations | 86.59% | 63.69% | 70.95% |
| Mean required-session recall @8 | 78.84% | 77.03% | 81.89% |
| All required sessions @8 | 138/200 | 129/200 | 145/200 |
| Mean reciprocal rank | 0.7952 | 0.7734 | 0.8075 |

The trained model rescued ten lexical session-level misses and regressed on
seven, for a net gain of three. At the exact-turn level it rescued fifteen but
regressed on forty-three, for a net loss of twenty-eight. Its fold-level
session hit@8 counts were `35, 36, 37, 35, 38` out of forty. In the canonical
rerun, warm reranking of all 96 candidates measured 61.14 ms p50 and 75.62 ms
p95. These are local-machine timings and varied from the superseded run, while
the rankings reproduced exactly.

All seven advance checks failed:

- 181/200 session hit@8 was below 190/200;
- 81.89% mean source recall was below 95%;
- 127/179 conditional exact-turn hit@8 was below 171/179;
- seven session regressions and forty-three exact-turn regressions both
  exceeded their caps of two;
- three folds were below 37/40; and
- 75.62 ms warm p95 exceeded 50 ms.

## Interpretation and decision

The learned scorer did not simply fail to learn. Relative to base MiniLM it
added four session-level successes without a regression, increased
all-required-session coverage from 129 to 145 questions, and increased exact
turn containment from 114 to 127. The objective therefore taught the intended
broad source-routing behavior.

The failure is compositional: replacing lexical order with the semantic score
throws away a strong exact-turn signal. The broad grade-1 set rewards any turn
from the correct answer session, so the source-balanced objective can improve
while decisive-turn containment falls. The largest pattern is consistent with
that trade: preference and temporal questions gained broad coverage, while
knowledge-update and single-session-user questions lost exact precision.

A post-hoc protected-union projection makes the division of labor concrete.
Taking lexical top eight plus deduplicated trained-MiniLM top eight preserves
all lexical successes and adds the distinct trained rescues, yielding 188/200
questions with a required session and 170/179 reachable exact annotations.
The union contains 13.185 memories on average, median 13, p95 15, and maximum
16. This is not a preregistered arm and still misses the respective frozen
thresholds by two questions and one question; it is design evidence only.

This generic MiniLM replacement arm is retired. It does not justify opening
validation100, confirmation200, or changing production. The next defensible
experiment is a residual/gated composition that preserves lexical position or
score as a protected exact-evidence signal and lets the learned scorer add
source diversity only when it can do so without evicting lexical evidence.
That experiment needs frozen no-regression accounting on both session and
exact-turn containment; merely increasing the exact-loss weight would be an
uncontrolled hyperparameter response on analysis-used development data.
