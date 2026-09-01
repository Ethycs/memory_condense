# Reduced specialist workload isolates technique failures

**Date:** 2026-08-28

**Status:** sealed post-hoc exact-ten specialist Terra/Sol diagnostic complete;
1/10 after validation; memory pressure rejected as the cause; not a locked-100
score

## Question tested

The exact-ten failures could still have had two qualitatively different causes:

1. the million-token stores, simultaneous namespace residency, or final prompt
   budget might be making otherwise adequate retrieval fail; or
2. the retrieval, composition, and answer-validation techniques might be
   mishandling evidence even when the working prompt is small.

This assay holds the previously ingested stores and known-miss population fixed
and answers only ordinals `7, 31, 36, 43, 61, 72, 77, 81, 86, 93`. It does not
re-ingest or replay a million-token transcript on the answer path. The protected
parent contribution is composed with independently selected numeric,
profile/preference, and temporal/insufficiency contributions. Each mechanism
receives a non-borrowable minimum, shared surplus is allocated afterward, exact
span duplicates are removed only after independent selection, and one terminal
fitter enforces the 8,000-token complete-envelope cap.

This is a post-hoc population chosen from known misses. Its score cannot be
added to the official 73/100 result and is not evidence for a deployable
question router.

## Workload and firewalls

- The answer plane reads the sealed specialist construction only. It never
  opens benchmark references or the post-hoc target audit.
- There is no million-token re-ingest, namespace-index replay, or persisted
  transformer token state on the answer path.
- The ten complete prompt envelopes range from 2,772 to 5,832 tokens, leaving
  at least 2,168 tokens below the 8,000-token cap.
- Terra receives ten unique dated-question plus sealed-memory prompts.
- The answer run and byte-identical replay are verified before the judge opens
  gold.
- Sol receives ten standard dated-question, reference, and sealed-prediction
  prompts. Judge materialization and replay are checkpoint-only.
- The run made exactly 10 Terra and 10 Sol physical calls. All materializers
  and replays made zero calls.

The first sandboxed Terra attempt was network-blocked after reserving four
request journals in `reduced-specialist-answer-v2`. Those ambiguous reservations
were preserved. The canonical run uses the fresh
`reduced-specialist-answer-v2-exec` root; it does not retry or reuse any of the
abandoned request identities.

## Structural result

The final fitted prompts retain 21/23 labelled source targets and are
source-set complete on 8/10 questions. Selection and terminal counts are
identical: the final fitter drops none of those 21 targets.

| Ordinal | Terminal targets | Complete envelope | Structural diagnosis |
| ---: | ---: | ---: | --- |
| 7 | 2/3 | 4,725 | missing sports-event span before composition |
| 31 | 2/2 | 4,576 | complete numeric operands |
| 36 | 1/1 | 4,337 | complete, answer-bearing comedian preference cluster |
| 43 | 2/2 | 5,438 | complete target and temporal comparator |
| 61 | 4/4 | 5,203 | complete furniture-operation set |
| 72 | 2/2 | 5,306 | tomato support plus sealed chili-count absence certificate |
| 77 | 3/3 | 5,595 | labelled sources complete; interval computation still wrong |
| 81 | 0/1 | 2,772 | cocktail preference never admitted |
| 86 | 3/3 | 4,679 | complete three-trip temporal bundle |
| 93 | 2/2 | 5,832 | target and comparator present, but global timeline over-expanded |

Source-ID reach remains weaker than answer-bearing span reach. For q7, the
temporal specialist's apparent source hit is a same-session alias attached to
an unrelated waterfront-path span. The parent contains the actual triathlon
and soccer events, while the 5K event is absent. Thus the temporal specialist
has zero answer-bearing sports spans even though the source audit reports one
of three for that method.

## Live answer result

The validated path scores **1/10** under Sol. Normalized exact match is 0/10
and mean F1 is `0.1116219766`. Terra changed only two protected-parent
predictions; Sol accepts only q93.

| Ordinal | Terra raw decision | Final sealed prediction behavior | Sol | Failure layer |
| ---: | --- | --- | --- | --- |
| 7 | keep wrong parent | unchanged | incorrect | sports-event admission |
| 31 | replace with `70 pounds` | correct raw answer rejected; restored `50 pounds` | incorrect | aggregate validator scope |
| 36 | replace with *The OA* | wrong replacement accepted | incorrect | cross-lane synthesis/salience |
| 43 | replace with `12 new tomato saplings` | correct raw answer rejected; restored peace-lily answer | incorrect | global deterministic advisory scope |
| 61 | replace with `4 pieces of furniture` | correct raw answer rejected; restored `1` | incorrect | aggregate validator scope |
| 72 | keep `5 tomato plants` | ignores the sealed missing-chili certificate | incorrect | insufficiency answer adapter |
| 77 | keep `0 months` | scalar advisory also computes the wrong `7 months`; gold is 5 | incorrect | temporal entity/interval selection |
| 81 | generic cocktail replacement | rejected, then wrong parent restored | incorrect | preference-domain admission |
| 86 | correct three-trip ordered list | correct raw answer rejected; restored one-trip fallback | incorrect | typed order validator |
| 93 | replace with a 20-event global timeline | noisy answer accepted because it contains the first-client contract | **correct** | global-to-local scope leak |

Four of the nine judged errors therefore contain the correct answer in Terra's
raw completion but lose it at deterministic validation: q31, q43, q61, and
q86. Conversely, q36's irrelevant recommendation passes validation, and q93's
twenty-event global timeline passes because it includes the requested fact.
The 1/10 score measures the whole answer stack; it is not a retrieval-only
score.

## Causal conclusion

This reduced workload rejects memory management as the operative recall cause:

1. the earlier resident-versus-one-namespace-per-child control reproduced all
   70 method/question outputs byte-for-byte while reducing simultaneous indexed
   tokens by 85.66%;
2. the present answer prompts are only 2.8k--5.8k tokens, and final fitting
   drops zero selected target sources; and
3. failures persist, including four cases where the LLM states the correct
   answer before the validator discards it.

The defects are technique-specific and fall into three groups.

### 1. Admission and linking

- q7 is routed correctly as temporal order, and all three memories exist in
  the immutable store. The specialist has no sports-event domain family and
  degenerates to the residual stem `dur`, selecting unrelated uses of
  "during". Add a bounded sports family covering run/5K/race/triathlon,
  soccer/tournament/league/game/match, and prefer completed participation
  assertions.
- q81 is routed correctly as synthesis, and the exact cocktail cluster exists.
  The profile gate recognizes singular `suggestion` but not plural
  `suggestions`, then has no beverage/cocktail domain. Accept plural
  recommendation cues and add a dedicated cocktail/drink/gin domain rather
  than broadening food.

### 2. Specialist-to-validator scope

The additive composer preserves parent and specialist evidence, but the legacy
validator recompiles completeness and deterministic execution over the entire
union. When no reliable action or slot scope exists, `count_or_aggregate`
requires every semantic unit in the prompt; this rejects q31 and q61 despite
correct specialist answers. The same global scope expands q43 and q93 into
irrelevant chronological histories and makes q86's correct ordered list fail
typed entailment.

The validator must consume the selected specialist's explicit operand groups,
temporal bundle handles, winner/predecessor handles, and absence certificate.
It must not infer the operator proof universe again from every parent handle.

### 3. Answer synthesis

Even with the exact q36 preference cluster present, the shared prompt lets a
parent screen-preference distractor dominate and produces *The OA*. q72 exposes
the inverse problem: a correct negative-evidence certificate is present, but no
answer adapter turns it into an insufficiency response. q77's selected interval
anchors yield neither the correct direct answer nor a correct scalar advisory.

The next treatment should send the LLM a small operator-scoped payload: the
specialist proof first, its exact raw chunk where useful, and the parent only as
a fallback. The final validator should check that same scoped proof. This keeps
the cumulative memory architecture while preventing global evidence from
silently redefining a local operator.

## Sealed canonical artifacts

| Artifact | SHA-256 |
| --- | --- |
| specialist construction | `fd179de8fc383cb6c051f704d5f0d25a37c93e3cb086ada794b3200ca89ada05` |
| post-hoc source audit | `f92beac86d144f49b6964200616b1fa0a6e6850c73cbcb67d8224214f7575b69` |
| Terra preflight | `1f151d31ae49a38979e9b1073d53a5323f13365d3bd398a2b069500c542d1bd7` |
| Terra answer and byte-identical replay | `7c0922377517c7311286d6e13dbdf2893db0de64481662afc6b0b71211ae14a5` |
| Sol preflight | `4bc6d198e79dd5a602b8acd744205b8efd55e95e89572c402bb0f7e4939c85e6` |
| Sol judge and byte-identical replay | `d913c31f07fe475fd94fb83e89afb9e1bd7824ff10883ed4622cb36879f8a670` |
| Sol score and byte-identical replay | `a1f709068b9b8bc075523cb8472b2c76d9e856ae6cfc34063266bb7b5e3745b2` |

Canonical answer root:
`eval_results/matched_eval_100/reduced-specialist-answer-v2-exec`.
Canonical judge root:
`eval_results/matched_eval_100/reduced-specialist-sol-judge-v2-exec`.

Focused retrieval, answer, and judge gates pass: `33 passed, 1 skipped`. The
skip is the expected optional environment-dependent smoke; the only warning is
the existing read-only pytest-cache warning.

## Next falsifiable step

Implement the two domain-admission repairs and an operator-scoped completion
contract, then rerun only this exact-ten diagnostic. The immediate gate is:

- q7 and q81 must gain answer-bearing spans before packing;
- q31, q43, q61, and q86 must preserve their already-correct raw replacements;
- q72 must deterministically emit insufficiency from its sealed certificate;
- q77 must use the correct entity-constrained interval anchors; and
- q93 must return the local milestone rather than a global timeline.

Only after those mechanisms pass should they be generalized and evaluated on
the full locked 100. No exact-ten result, however high, is promotable as the
official score.
