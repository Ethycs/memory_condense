# Gate-derived full100 and confirmation200 terminal promotion

**Date:** 2026-08-29
**Status:** architecture assay only; no provider calls, answer run, judge run, or score claim

## Purpose

The exact-11 P/R/L/G terminal assay establishes a bounded implementation and
replay boundary, but it is not itself a production routing rule. This note
defines the minimal path for promoting that frozen terminal policy first to
the complete locked validation100 population and, only after the validation
gate passes, to the untouched confirmation200 population.

The central constraint is that production routing remains question-local and
gold-blind. An ordinal list may describe a sealed population after the gate
has run, but it must never decide which rows receive terminal retrieval.

## Sealed validation eligibility

The authoritative validation gate is:

`eval_results/matched_eval_100/locked-semantic-residual-v4-r7/locked-semantic-residual-gate-v4.json`

Its file SHA-256 is
`779c711e090ecb9faad92d9845158d939411dfa3a965669a26cfe8a8062fb912`.
It binds the R7 construction with SHA-256
`d0f226b1577a6bf40c54758d2fdc477ab98483613ca7c4fc77ef93383a651f6a`.

The sealed population is:

| Gate result | Rows | R7 construction mode |
|---|---:|---|
| generically residual-eligible | 68 | `residual_synthesis` |
| noneligible | 32 | `not_eligible` |
| total | 100 | complete locked population |

All ten validation namespaces contain at least one eligible row. The gate
uses only sealed question, answer, route, frontier, sufficiency, specialist,
and reconciliation state. It accepts no reference answer, correctness label,
judge result, target question ID, or target ordinal.

For population auditing only, the 68 derived zero-based ordinals are:

`3, 4, 5, 6, 7, 9, 11, 13, 14, 18, 19, 20, 21, 22, 24, 27, 28, 29, 30, 31, 32, 33, 35, 36, 40, 41, 43, 46, 47, 48, 49, 50, 51, 53, 54, 56, 57, 58, 59, 61, 63, 64, 65, 66, 67, 68, 69, 70, 72, 73, 75, 77, 78, 81, 82, 83, 84, 85, 86, 88, 91, 92, 93, 94, 95, 97, 98, 99`.

This list is an observed receipt-bound projection, not a policy input. The
full100 runner must reconstruct it from the sealed eligibility decisions.

## Minimal full100 construction

Promotion should create a new full100 artifact lineage rather than modify the
frozen exact-11 construction or its default bytes. The new driver should:

1. authenticate the exact R7 gate, construction, query-vector replay, V3
   parent answer, V6 source-local result, V7 global result, and terminal policy;
2. derive the work population from `eligible == true` together with R7 mode
   `residual_synthesis`;
3. execute the unchanged cumulative P/R/L/G terminal compiler for those rows;
4. process the ten resident namespace indexes once each;
5. emit one ordered row for every ordinal from 0 through 99; and
6. construct and replay with zero provider calls and zero retained transformer
   token state.

The 100-row construction must have two explicit modes:

| Population | Terminal action | Answer-stage behavior |
|---|---|---|
| 68 eligible rows | compile the frozen P/R/L/G prompt | one logical terminal answer |
| 32 noneligible rows | no terminal compilation or prompt | authenticated V3 passthrough |

The derived eligible ordinals and their population receipt may be recorded for
replay auditing. They must not appear as an executable allowlist, CLI routing
rule, or source-ID exception.

## Passthrough and fallback invariants

The gate's `current_prediction` matches the sealed V3 prediction on all 100
rows. Consequently, every noneligible row must preserve the V3 prediction
byte-for-byte and bind its exact V3 source-row identity. Such a row receives no
terminal prompt and consumes no Terra call.

No row may disappear because construction or parsing fails. An eligible row
whose provider-free terminal construction is unavailable must carry an
explicit construction fallback to V3. An eligible row with a malformed,
ungrounded, or otherwise invalid completion must use the existing immediate
V3 parent fallback. Both cases remain visible in the 100-row answer artifact
and in the full-population judge input.

Selection and evidence deduplication remain unchanged: each P/R/L/G lane
selects under its independent budget before protected-owner deduplication, and
the terminal compiler retains exact provenance and owner substitution.

## Answer and judge lifecycle

Assuming all 68 eligible rows produce valid terminal plans, the validation
lifecycle is:

| Stage | Logical population | Default new provider calls |
|---|---:|---:|
| full100 construction and replay | 100 rows / 10 namespaces | 0 |
| Terra terminal answer | 68 prompts | 68 |
| answer materialization and replay | 100 rows | 0 |
| Sol semantic judge | 100 prompts | 100 |
| judge and score materialization/replay | 100 rows | 0 |

All 68 eligible rows require a logical reanswer because their terminal
P/R/L/G message identity differs from the earlier R7 answer message. The
official score must judge all 100 final predictions, not only changed rows,
eligible rows, or the exact-11 subset.

Exact-11 checkpoint reuse is optional, not assumed. It is valid only when the
full100 prompt has the identical message hash, model/request policy, response
journal, and completion receipt, and when the checkpoint is imported through
an explicitly authenticated source binding. Under those conditions, eleven
checkpoint hits could reduce new Terra calls from 68 to 57 while leaving the
logical prompt population at 68. Merely matching an ordinal, prediction, or
evidence set is insufficient. Without that proof, the default budget remains
68 new Terra calls.

The answer artifact therefore needs 68 physical-prompt plans plus 32 explicit
passthrough plans, followed by a merged, ordered 100-row run and byte-identical
replay. The judge artifact should use the existing full100 typed-final judging
contract rather than the exact-11 selected-subset wrapper. Reusing earlier V3
judge verdicts would require a separate mixed-judge identity proof and is not
the minimal release path.

## Confirmation200 promotion boundary

Confirmation remains closed until the unchanged full100 terminal policy earns
at least 95/100 and its routing, budgets, answerer, and judge are frozen. The
same question-local gate must then be applied to confirmation; the validation
68/32 split, eligibility ordinals, and question identities cannot be reused.

The production firebreak already fixes the confirmation population to 200
rows and can export a closed-schema, label-free treatment projection containing
only histories, source coordinates, timestamps, question text, and question
date. The locked claim profile uses deterministic ten-question approximately
one-million-token shards, so confirmation requires 20 namespace/store passes.
These are 20 distinct execution namespaces, each retaining the same locked
ten-question shard boundary.

If `E_c` is the number of confirmation rows admitted by the frozen generic
gate, the terminal-stage call plan is:

- zero provider calls for firebreak export, store/index construction, terminal
  construction, and replay;
- `E_c` logical Terra terminal-answer calls, with `200 - E_c` authenticated
  parent passthrough rows; and
- 200 Sol calls for the official full-population judge.

`E_c` is intentionally unknown until the label-free upstream confirmation
construction runs. Provider calls needed to construct the frozen upstream
parent-answer lineage are additional and must be authorized from its own
generic gate receipts; they cannot be estimated by copying validation rates.

## Confirmation prerequisites and current gaps

The raw 277,383,467-byte dataset, split manifest, population locks, and
confirmation firebreak implementation exist. The confirmation memory
experiment does not yet have the artifacts required by P/R/L/G:

- a no-clobber confirmation treatment export and receipt;
- twenty ten-question retrieval shards, query inputs, source stores, and
  combined approximately one-million-token stores;
- source-vector and query-vector construction/replay;
- question-neutral residual-tree and global semantic indexes;
- exact per-namespace episode-artifact bindings;
- confirmation-specific upstream V2/V3 parent answers and status receipts;
- a confirmation eligibility gate derived from those runtime states;
- P/R protected-owner, L source-local, and G global construction/replay;
- terminal answer checkpoints and full200 answer replay; and
- confirmation judge and score artifacts.

Therefore, confirmation cannot be evaluated faithfully by pointing the
exact-11 or validation100 runner at new ordinals. The complete upstream generic
baseline and its memory artifacts must first be reconstructed from the sealed
label-free confirmation treatment input. References and gold remain outside
that process and are opened only after all 200 answers are frozen.

## Required confirmation report

The final report must publish both:

1. the predeclared result over all 200 confirmation rows; and
2. the sensitivity result over the 185 rows not named in the production
   exposure ledger.

It must disclose that 15 confirmation answers were potentially exposed in
answer-only metadata. Neither the full200 nor the non-exposed185 slice may be
used to retune the already frozen policy.

This note records an execution architecture and population contract only. It
does not establish any full100 improvement, 95% result, or confirmation score.
