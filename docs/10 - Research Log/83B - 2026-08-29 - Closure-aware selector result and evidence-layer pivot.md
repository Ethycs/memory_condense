# Closure-aware selector result and evidence-layer pivot

**Date:** 2026-08-29

**Status:** complete replay-verified development experiment; V5 scores
**88/100** and is rejected as an answer parent. V3 remains the protected
**89/100** baseline. V5's closure/search receipts remain useful routing input
for the source-local and global evidence layers.

## Question

R7 found useful bounded evidence but its lexical replacement validator both
rejected grounded syntheses and admitted two regressions. V5 tested whether a
gold-blind Sol verifier could safely choose between each exact raw Terra
replacement candidate and the exact protected V3 answer, without generating
or rewriting either string.

## Frozen construction

V5 authenticated the exact R7 preflight, answer, replay, provider journals,
and V3 parent before constructing any selector prompt. It mechanically froze
all 15 raw R7 `replace` candidates and separately receipted 13 exact-current
responses whose only protocol defect was carrying evidence handles. The other
40 residual responses were canonical keeps; 32 questions were already V3
passthroughs.

The provider-free preflight had these properties:

| Property | Result |
| --- | ---: |
| unique selector prompts | 15/15 |
| no-call keep normalizations | 13 |
| maximum complete selector envelope | 7,742/8,000 tokens |
| maximum serialized R plane | 2,393/2,400 tokens |
| maximum serialized P plane | 793/2,400 tokens |
| maximum full enriched R/P union | 4,581 tokens |
| deterministic open-frontier search triggers | 11/15 |
| construction provider calls | 0 |
| retained transformer-token state | 0 bytes |

R and P remain independent non-borrowable planes. Every final V5 prediction
is byte-identical to either the protected V3 prediction or the exact sealed
Terra candidate. The verifier cannot write, merge, or edit answer text.

An early implementation briefly treated two locally executable operands as a
scoped closure proof. Adversarial review rejected that shortcut: observing two
numbers does not bind them to every required entity, temporal relation, or
open-world alternative. The sealed policy instead routes every open R7 row
whose question-only `TypedOperatorSpec` requires a complete frontier to
current plus `needs_global_search`. Arithmetic execution proves only the
calculation, never retrieval completeness.

## Provider and replay result

The selector made exactly 15 Sol calls with zero retries and zero checkpoint
hits. Checkpoint-only materialization selected three candidates, changed three
V3 predictions, and emitted eleven deterministic search triggers. Exact
replay made zero calls and reproduced identical bytes.

The selected ordinals were 36, 49, and 81. Ordinal 6 was classified as an
equivalent paraphrase and canonicalized to current. The search-triggered
ordinals were 14, 18, 31, 32, 50, 51, 56, 69, 77, 97, and 98. This routing is
derived from question-only operator/frontier state, not an ordinal policy.

The independent judge then authenticated the byte-identical V5 answer run and
replay before opening validation gold. It sealed 100 unique Sol prompts and
made exactly 100 physical calls with zero retries and zero checkpoint hits.
Judge and score materialization used checkpoints only; both replayed
byte-identically with zero calls.

| Artifact | SHA-256 |
| --- | --- |
| V5 selector preflight | `7281bc758e37013821b3589985f786aef74adc5eed884cb358039c4ff290b86f` |
| V5 answer and byte-identical replay | `3645a869bbee3835f1e9bc3a8c1d7104738bbbcd5266c319fa23138e77ed02c7` |
| V5 Sol judge preflight | `1a18b873f3f2887fec4f1fafaff99fbccedc45ce4308a834ce6ac1512687f12d` |
| V5 Sol judge and byte-identical replay | `48dad095ed055efbb6efea8dbc7b47b63a7cc3283e79032f117b816a9c1fa30b` |
| V5 Sol score and byte-identical replay | `3d7ce88e66e97b32c28b83649e509b601f64c49d3eec4dc78be4aa6f6f7a6732` |

## Accuracy result

V5 scores **88/100**, with normalized exact match 32/100 and mean normalized
F1 0.597753. The judged misses are:

`14, 28, 36, 40, 49, 53, 54, 67, 69, 82, 94, 97`.

Relative to V3:

- ordinal 36 changes from correct to incorrect;
- ordinal 49 changes but remains incorrect;
- ordinal 81 changes and remains correct; and
- every unchanged V3 answer receives the same verdict as in V3's judge.

V5 therefore has zero rescues, one regression, and a net movement from 89 to
88. Relative to R7, it correctly restores ordinals 31 and 51 to their V3
answers, but ordinal 36 regresses and the fresh judge returns byte-identical
ordinal 82 to incorrect after R7's one-run judge flip. The net remains 88.

The semantic selector answered the narrow validation question: replacing the
lexical validator is safer for open counts and derived answers, but selecting
among the existing strings cannot repair missing answer-bearing evidence or a
candidate that omits the decisive named preference. V5 is therefore retained
as a routing and safety diagnostic, not promoted as the answer parent.

## Exact miss-flow assay

After scoring, an exact reconstruction replayed every R7 compact commitment
and located the answer-bearing development spans. This is contaminated
diagnostic evidence, not a production routing table.

- No miss lost its answer leaf during semantic-tree classification.
- Seven misses are ranking/packing or operand-closure failures: 14, 28, 40,
  53, 54, 67, and 69.
- Ordinal 49 reaches the right Denver/music source and evidence, but the
  candidate omits the decisive named preference and remains wrong.
- Ordinal 94 contains conflicting source dates and reference semantics.
- Ordinal 97 loses the personal UberEats comparator, while the surviving
  record still does not establish the requested first-order percentage.

Representative complete retained-segment ranks show the connectivity problem:
the missing smoker assertion ranks 643; the second bike operand ranks 1,864;
the jewelry assertions rank 858, 885, and 1,015; the two museum assertions
rank 45 and 86; and the missing blazer action ranks 103. These leaves survive,
but ordinary global ranking cannot fit them into the 2,400-token R plane.

## Decision

Keep V3 as the protected answer population. Continue cumulatively:

1. use V5's generic closure/search receipts to identify unresolved work;
2. reopen exact selected source groups and episode neighbors in a separately
   budgeted V6 L plane, promoting user assertions over generic source tails;
3. run V7 global typed semantic search for unresolved sources and operands;
4. select before EM/protected dedup, reinject every exact owner, and keep the
   frontier open whenever slot or operand closure is unproved; and
5. synthesize only after the answer-bearing exact spans fit under the terminal
   8,000-token envelope.

No confirmation claim follows from this result. Validation100 is already
analysis-used; any eventual >=95 policy must be frozen before confirmation200
is materialized, and the fair Mem0 arm remains independently pending.
