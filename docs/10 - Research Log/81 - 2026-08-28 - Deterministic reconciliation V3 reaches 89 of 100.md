# Deterministic reconciliation V3 reaches 89 of 100

**Date:** 2026-08-28

**Status:** the provider-free temporal, numeric, and cross-plane authority
reconciliation stack is sealed and replayed; an independent 100-call Sol judge
scores the resulting full population **89/100**, up from the sealed V2 result
of **79/100**. The 95% target remains unmet. This validation population has
now been used for failure analysis and policy development, so 89/100 is a
development result, not an uncontaminated confirmation claim.

## Outcome

V3 changes exactly ten of the 100 sealed V2 predictions. All ten changes turn
V2 judge failures into V3 judge successes, and no previously correct answer
regresses:

| Lane | Ordinals | V3 outputs |
| --- | --- | --- |
| question-bound temporal | 15, 16, 17, 25 | about three months; three months; smart thermostat; about one week |
| sealed numeric | 34, 44, 87, 90 | 3; 4; 5 fitness classes per week; 32 |
| cross-plane parent protection | 12, 76 | 15; 5 |
| byte-preserved V2 fallback | remaining 90 | unchanged |

The result is therefore an observed, independently judged **79 -> 89** gain,
not an arithmetic projection from a reduced subset. Normalized exact match is
32/100 and mean token F1 is `0.5994792110806573`; semantic Sol judgment remains
the registered answer metric.

The 11 remaining Sol failures are ordinals `14, 28, 40, 49, 53, 54, 67, 69,
82, 94, 97`. They are now the development residual. Ordinal 94 retains a known
benchmark date/reference inconsistency and must not be used to weaken the
generic runtime policy.

## Composition contract

V3 reads only the exact sealed V2 preflight, terminal answer run, and
byte-identical replay. It opens no gold and reaches no completion provider. It
re-executes the lanes in this frozen order:

1. question-bound temporal reconciliation on temporal routes;
2. sealed numeric reconciliation on numeric routes;
3. cross-plane protection of a stronger parent on validated replacements; and
4. byte-preserved V2 fallback.

Materialization requires the freshly recomputed full-72 lane status population
to equal these frozen receipts:

| Lane | Full-72 status population SHA-256 |
| --- | --- |
| temporal | `a8a8d5c4ed538056cc2a6376b6839b61b6684bdbd821b8329ef6eb41d66f89a6` |
| numeric | `a8463c0320c5cf0840ec881ffd1c418b979ed52bd782c335a7fad5ccd3876ed0` |
| authority | `2970a7daa65b8a30003a5ecd237f75172fb1ee44c2263bb9516842075f3f698c` |

A valid-looking but wrong SHA is rejected. The composition policy receipt is
`774af3e60853d67ada1b58850389c4a9339ca4cecaacd46895febd5626b417be`.
The maximum inherited complete provider envelope is **7,481/8,000** tokens.
V3 itself makes zero provider calls and retains zero transformer-token state.

The actual-schema numeric V2 assay is separately sealed at
[the numeric population](../../eval_results/matched_eval_100/numeric-evidence-reconciler-v2/locked-specialist-final-answer-v2-numeric-population-v2.json).
It contains 72 rows: four supported, 62 insufficient, and six conflicted. Its
population receipt is
`87bb0862e00727be3387fdab39854500fbe74165b84290c3147d83b355b17038`,
top-level receipt is
`cbd7c02e24ca824c5ec01c64b59c56d14ccecb14dfa5b7d443cdcf38be03a8aa`,
and sealed artifact SHA is
`108cdaf00488ecb5fdf205d45ec5ea452312369975e11556ec9361d36b5a6952`.
It preserves V1's q44/q90 decisions and adds only q34 and q87. Q87 binds a
base frequency of three plus Hip Hop and BodyPump once each; the second
BodyPump observation is receipt-bound corroboration, not another class.

## Sealed answer and judge artifacts

| Artifact | SHA-256 |
| --- | --- |
| [V3 answer run](../../eval_results/matched_eval_100/locked-specialist-final-reconciliation-v3/locked-specialist-final-reconciliation-v3.json) and byte-identical replay | `07c6f3125e65094880384c1c1c6f7d9be0600475f1fe58d050796fc0f48493d1` |
| [Sol preflight](../../eval_results/matched_eval_100/locked-specialist-final-reconciliation-sol-judge-v3/locked-specialist-final-reconciliation-sol-judge-preflight-v3.json) | `5513ece9387802d03b2f5b637832f7204a36de66fab9387c37f192386a218988` |
| [Sol judgment](../../eval_results/matched_eval_100/locked-specialist-final-reconciliation-sol-judge-v3/locked-specialist-final-reconciliation-semantic-judge-sol-v3.json) and replay | `ce8fad414d87c21428f1264c0b7790cf261979642ec930802674bfba29077bf5` |
| [score](../../eval_results/matched_eval_100/locked-specialist-final-reconciliation-sol-judge-v3/locked-specialist-final-reconciliation-score-v3.json) and replay | `ffa89e128afa9500358c7b259934d9c8c24437eba7422a4853487583d517a86d` |

The Sol stage made exactly 100 physical calls with zero retries. Judgment and
score materialization then used 100 response checkpoints, made zero further
calls, and replayed byte-identically. The combined answer/lane/base regression
suite passed 72 tests; the numeric V2 and its runner passed 30 tests; the
base/V2/V3 judge adapters passed 13 tests.

## What this establishes, and what it does not

The ten-for-ten delta establishes that the principal V2 loss was not missing
million-token ingest. The relevant compact evidence or protected parent was
already present, but the final projection layer lacked the right temporal,
numeric, or cross-plane authority operator. Deterministic, receipt-bound
operators repair that seam without changing retrieval or asking another LLM.

It does **not** establish 95% generalization. The next layer is the bounded
semantic residual path over the V3 fallback population: a separately budgeted
semantic binary search, post-selection deduplication against protected EM
evidence, and a bounded Terra synthesis call. Its eligibility population and
construction must freeze before opening the V3 score. After the final policy
freezes, the target must be evaluated on the untouched 200-question
confirmation partition. The fair Mem0 arm also remains pending and must use
the same corpus, question order, prompt cap, Terra reader, and Sol judge.
