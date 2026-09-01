# Scoped specialist repairs recover the reduced exact ten

**Date:** 2026-08-28

**Status:** sealed post-hoc exact-ten Terra/Sol treatment complete; 10/10
semantic judge accuracy; memory pressure rejected for this miss set; not a
locked-100 score

## Question tested

Research Log 78 reduced the workload to the ten previously missed ordinals
`7, 31, 36, 43, 61, 72, 77, 81, 86, 93`. Its 2.8k--5.8k-token prompts still
scored 1/10 because the stack either failed to admit the decisive local span
or validated a specialist answer against the whole parent evidence union.

This treatment asks whether general, question-typed repairs can recover those
same ten failures while leaving the immutable long-chat stores, protected
parent contribution, hard 8,000-token complete-envelope cap, exact
provenance, and zero persisted transformer token state intact.

The population is deliberately post-hoc: all ten questions were selected
because an earlier system missed them, and the repairs were developed from
their failure modes. Therefore 10/10 here is a treatment-development result.
It cannot be added to the official 73/100 score, relabelled as 83/100, or used
as evidence that a deployable router will generalize.

## Repairs under test

The cumulative memory architecture remains additive. The changes are local to
specialist admission, representative choice, and proof-scoped synthesis:

1. Temporal order expands a sports-event domain, prefers completed
   participation assertions, and chooses the answer-bearing event sentence
   over an adjacent same-session discussion.
2. Temporal interval applies the requested entity and participant constraints
   before selecting its start event, then computes from that winner to the
   query timestamp. Later near-matches are bounded comparators, not interval
   endpoints.
3. Profile retrieval recognizes plural recommendation/suggestion language and
   has a beverage/cocktail domain. A coherent intrinsic preference cluster
   outranks a request-only cluster.
4. The answer model receives a closed specialist proof scope. Numeric groups,
   temporal order, temporal winner, interval arithmetic, profile clusters,
   and absence certificates each have a distinct deterministic validator.
   Parent evidence remains a fallback but cannot expand the proof universe.
5. Temporal-relative and interval answers may cite only the winner handle;
   predecessor handles remain internal comparators. This prevents a model
   from reconstructing a global timeline through proof citations.
6. The completion runtime fails closed when a locally counted response exceeds
   the configured 768-token output reserve, both before checkpoint publication
   and during checkpoint replay.

These rules are gold-blind on the answer plane. The post-hoc source audit opens
the target plan only after the construction has been sealed and verified.

## Provider-free structural gate

The canonical construction reaches **23/23 labelled source targets** and is
source-set complete on **10/10 questions**. Final fitting loses none of those
targets. All ten specialist advisories are nonempty.

Complete prompt envelopes range from **4,213 to 5,849 tokens**, including the
768-token output reserve. The largest envelope leaves 2,151 tokens below the
8,000-token cap. No million-token transcript is submitted to Terra: the
existing immutable approximately-1M-token namespace stores are scanned
locally, then only the selected proof-carrying payload is rendered.

Three previously diagnostic cases now contain the exact local evidence:

- q7 orders the June 2 Spring Sprint Triathlon, June 10 Midsummer 5K, and
  June 17 company charity soccer participation assertion. The adjacent
  hydration question is absent from the specialist bundle.
- q77 selects the October 22 Science Museum visit with one friend as the
  interval start, binds the question timestamp as the implicit end, and leaves
  predecessor null. The January, February, and March museum events are bounded
  comparators only.
- q81 carries the coherent Pimm's Cup, classic-cocktail, Hendrick's gin,
  cucumber, and citrus-simple-syrup preference/request cluster.

The final provider-free freeze gate is `86 passed, 1 skipped`. The skip is the
expected optional environment-dependent smoke; the only warning is the known
read-only pytest-cache warning.

## Live answer and judge result

The answer path made exactly **10 Terra calls**. It emitted ten valid scoped
replacements, with no parent fallback, then replayed byte-identically from ten
checkpoints with zero calls. Across the batch, the rendered prompts total
41,723 proxy tokens and the visible completions total 315 proxy tokens.

| Ordinal | Scoped proof | Sealed Terra prediction | Sol |
| ---: | --- | --- | --- |
| 7 | temporal order | Spring Sprint Triathlon -> Midsummer 5K -> company charity soccer tournament | correct |
| 31 | numeric operand groups | 70 pounds | correct |
| 36 | profile preference | *Bo Burnham: Make Happy* | correct |
| 43 | temporal winner | 12 new tomato saplings | correct |
| 61 | numeric operand groups | 4 pieces of furniture | correct |
| 72 | absence certificate | 5 tomato plants; insufficient evidence for chili peppers | correct |
| 77 | temporal interval | 5 months | correct |
| 81 | profile preference | Hendrick's gin, cucumber, and citrus-syrup summer cocktail | correct |
| 86 | temporal order | Muir Woods -> Big Sur/Monterey -> Yosemite | correct |
| 93 | temporal winner | signed a contract with the first client | correct |

The judge path then made exactly **10 Sol calls** and scored **10/10 semantic
correct (100%)**. Judge materialization and replay use ten checkpoint hits and
zero provider calls; both judge and score replays are byte-identical.

The local lexical metrics remain much lower: normalized exact match is **1/10**
and mean token F1 is `0.516815434307694`. This is expected for concise semantic
equivalents such as `5 months` versus `5`, recommendation wording, and ordered
event names. The primary target defined in Research Log 06 is independent
answer-stage judge accuracy, not string identity.

## Causal interpretation

On the same post-hoc ten-question population, the earlier scoped-workload
precursor scored 1/10 and this repaired path scores 10/10. The answer prompts
remain small, final packing preserves every selected target, and the existing
long-chat stores are unchanged. This is strong evidence that these ten misses
were caused by technique rather than memory residency or prompt pressure:

- q7 and q81 were admission and within-source representative failures;
- q77 was entity-constrained local-to-global interval selection;
- q31, q43, q61, and q86 were correct specialist answers discarded by a
  validator whose scope was too global;
- q72 needed an explicit insufficiency answer path; and
- q36 and q93 needed one local proof cluster/winner to dominate unrelated
  parent context.

The result does **not** establish 95% on the locked 100. Because the treatment
was developed against known misses, the next honest test is to generalize the
mechanisms without ordinal-specific routing and run the complete locked
population once. Only that run can update the official score. A same-budget
Mem0 answer/judge arm also remains required for the preregistered comparison.

## Sealed canonical artifacts

| Artifact | SHA-256 |
| --- | --- |
| specialist v3 construction | `8edf924197fb1cb275b837272fe6583da055e42c98606ec6764d4ef93abc9e30` |
| post-hoc source audit | `5a3eca093e48e34afae8a395b35a80de67b7f056d0f9674b7a2312b4408ba93a` |
| Terra preflight | `55d24fd1f261f360370bbf4b818e53074bbdc0b08a47b0c7292153c137b4eb9f` |
| Terra answer and byte-identical replay | `6cf7297c6f79366ce59f1624c273dfb40295159a3dcdeafbd30c0921760d2380` |
| Sol preflight | `4a928e950f81668930a7a126edb7de524d6a7254250d0697c8446d201a92664d` |
| Sol judge and byte-identical replay | `951b2094b533351768e7f0f5cef32b4e387e87a3a376b07949f080f368959bc9` |
| Sol score and byte-identical replay | `0c69edc1deeaf362b8e9438bdf36390dd77eec57009596609343e19602fcf19c` |

Canonical roots:

- `eval_results/matched_eval_100/reduced-specialist-missing10-v3`
- `eval_results/matched_eval_100/reduced-specialist-answer-v3`
- `eval_results/matched_eval_100/reduced-specialist-sol-judge-v3`

## Next falsifiable step

Promote the general specialist ownership and scoped-proof contracts into the
locked-100 construction path without changing the population or consulting
gold. Preflight must prove every question has one deterministic owner/fallback
contract, a bounded prompt, and no retained transformer token state. Then run
one sealed 100-call Terra answer campaign and one sealed 100-call Sol judge
campaign. The target remains at least 95/100; this exact-ten result is a green
development gate, not the finish line.
