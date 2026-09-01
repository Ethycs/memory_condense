# Fail-closed terminal validator v4 revalidation

Date: 2026-08-30

## Result

The sealed R7 Terra outputs were revalidated without another provider call.
Validator v4 accepts the already returned q40 answer `3 pieces.` and q54 answer
`A smoker.`. The other nine exact-11 predictions remain byte-identical. The
result contains zero invalid completion fallbacks, zero new provider calls, and
zero retained transformer-token-state bytes.

- Historical v3 answer run:
  `a627ab7335282a46beee42a8d02e3becb6dc0cfc6a497d23f30b434b7d7b21e1`
- Historical v3 answer replay:
  `fbc6633ea1e6418b49c25cb20458ca5548589329c7af330ec8b4c07e24c8e7ab`
- Validator-v4 answer run:
  `9378772bce6d575660a27c243405d564cb03a5922e1b484f7818a4d730616d99`
- Validator-v4 byte-identical replay:
  `17901f9d5e6d55cefad38426d5ec2800314270963252beb67248deba1300b03f`
- Validator-v4 output root:
  `eval_results/matched_eval_100/locked-semantic-global-terminal-terra-answer-v2-r7-network-recovery1/validator-v4-r1`

The original provider population is unchanged. Its prompt-population SHA-256 is
`70b2317888ce6c0303f642da47ade7aa1a809d44c514e2761669d9ce6b09487b`.
The largest prompt is 7,226 proxy tokens; with the fixed 768-token output
reserve, the largest complete envelope is 7,994 of the hard 8,000-token cap.

## Why v3 rejected correct evidence-grounded answers

Q40 asked how many pieces of jewelry the user acquired in the last two months.
Its typed operator had operation `count_or_aggregate`, no compiled slots, and
only the canonical action `acquire`. V3 therefore expanded the completeness
universe to every retained row whose chunk contained an acquisition cue. That
included laptops, subscriptions, possible future jewelry purchases, repair
advice, and other distractors. Terra cited the three exact user acquisition
chunks, but v3 returned `aggregate_scope_incomplete` because it treated those
distractors as mandatory proof units.

Q54 asked what kitchen appliance the user bought ten days earlier. Its typed
operator had `temporal_mode=relative_select` but no executable temporal window.
The time executor skipped its relative-selection branch and fell through to
generic chronological ordering, then incorrectly labelled a concatenation of
all 27 dated summaries and all 27 handles as `supported`. V3 treated every
supported deterministic advisory as exclusive, so Terra's exact citation of
the March 15 smoker row was rejected as
`deterministic_advisory_disagreement`.

Both failures occurred after successful retrieval. They were validation-policy
false negatives, not missing evidence or prompt packing failures.

## Generic v4 repair

The repair contains no question ordinal, question ID, source ID, reference
answer, gold label, semantic-atom declaration, or expected answer literal.

1. `RELATIVE_SELECT` no longer falls through to generic timeline execution.
   When the v1 operator spec has no executable target, the deterministic
   executor returns `insufficient:relative_selection_target_unresolved`.
2. A deterministic advisory is binding only for a deterministic answer shape.
   Direct and synthesis shapes require the semantic candidate arbiter even if
   a lower-level time/state executor emitted a string.
3. Aggregate proof narrowing starts from question-derived terms only. A term
   must also occur in every cited semantic row before it can restrict the
   action-wide universe. A candidate-only term can never hide uncited rows.
4. When all cited rows prove the question action as completed, v4 restricts the
   proof universe to completed rows proving that same question action. For a
   first-person memory question, an all-user citation set may similarly remove
   assistant-authored advice.
5. Question-only temporal compilation recognizes bounded calendar lookbacks.
   It applies the query date, inclusive lower bound, future exclusion, and a
   conservative explicit month/day override when the selected exact chunk
   states an event date different from its source timestamp.
6. Exact relative-day answers must cite an answering row on the derived target
   calendar day. For q54, March 25 minus ten days is March 15. A toaster from
   March 18 and mixed-date citations fail closed in adversarial tests.

The q40 scope consequently contains the three completed, user-authored,
in-window jewelry-acquisition assertions and still rejects one- or two-row
omissions, a laptop-only citation, and a mixed jewelry-plus-laptop proof. Q54
accepts the smoker citation and rejects wrong-day and mixed-day alternatives.
Executable numeric advisories remain exclusive, so v4 is not a general
relaxation of deterministic agreement.

## Immutable version and contamination boundary

The historical v3 run and replay are not edited or reinterpreted in place.
Validator v4 derives a new contract from each exact v3 contract plus the exact
dated question already present in the sealed provider prompt. Every v4 decision
binds both the legacy-contract identity and the derived v4-contract identity.
The provider-free lifecycle authenticates the original preflight, answer run,
answer replay, post-seal gate, completion text, completion hash, prompt hash,
call key, request journal, and response journal before publishing a distinct
v4 run and replay.

This repair was developed after inspecting the q40/q54 v3 rejection codes and
sealed Terra completions. It is therefore post-hoc development evidence on the
exposed validation100/exact-11 population, not a new unbiased locked accuracy
estimate. The v4 policy must be frozen before confirmation evaluation. A score
from a fresh locked population, rather than these two repaired development
rows, is required for the 95% completion claim.

The validator itself remains gold-free and content-addressed. The separate Sol
judge adapter can opt into the replay-verified v4 answer source, but opening
reference answers and making judge calls remains a later, explicit evaluation
phase.

## Verification

The focused provider-free suite passed 36/36:

```powershell
.pixi\envs\dev\python.exe -m pytest -q `
  tests\test_matched_eval_typed_memory_final_validator_v4.py `
  tests\test_matched_eval_typed_operator_executor.py `
  tests\test_revalidate_locked_semantic_global_terminal_answer_v4.py `
  tests\test_run_locked_semantic_global_terminal_judge.py `
  -p no:cacheprovider `
  --basetemp .test-tmp\validator-v4-combined-author-20260830-a1
```

The sealed materialization and replay commands were:

```powershell
.pixi\envs\dev\python.exe `
  tools\revalidate_locked_semantic_global_terminal_answer_v4.py materialize `
  --answer-root eval_results\matched_eval_100\locked-semantic-global-terminal-terra-answer-v2-r7-network-recovery1 `
  --expected-answer-preflight-sha256 cbfca80dc2439ecf49e41f4207451eb829192dedde19cfa2a3214d8588ffb2d6 `
  --expected-answer-run-sha256 a627ab7335282a46beee42a8d02e3becb6dc0cfc6a497d23f30b434b7d7b21e1 `
  --expected-answer-replay-sha256 fbc6633ea1e6418b49c25cb20458ca5548589329c7af330ec8b4c07e24c8e7ab `
  --postseal-audit eval_results\matched_eval_100\locked-semantic-global-terminal-v2-r7\semantic-global-terminal-postseal-fact-audit-v2.json `
  --expected-postseal-audit-sha256 1618ba74541d21f21d60bc6a3464d141d113630065074cf6169b352aa9857663 `
  --output-root eval_results\matched_eval_100\locked-semantic-global-terminal-terra-answer-v2-r7-network-recovery1\validator-v4-r1

.pixi\envs\dev\python.exe `
  tools\revalidate_locked_semantic_global_terminal_answer_v4.py replay `
  --answer-root eval_results\matched_eval_100\locked-semantic-global-terminal-terra-answer-v2-r7-network-recovery1 `
  --expected-answer-preflight-sha256 cbfca80dc2439ecf49e41f4207451eb829192dedde19cfa2a3214d8588ffb2d6 `
  --expected-answer-run-sha256 a627ab7335282a46beee42a8d02e3becb6dc0cfc6a497d23f30b434b7d7b21e1 `
  --expected-answer-replay-sha256 fbc6633ea1e6418b49c25cb20458ca5548589329c7af330ec8b4c07e24c8e7ab `
  --postseal-audit eval_results\matched_eval_100\locked-semantic-global-terminal-v2-r7\semantic-global-terminal-postseal-fact-audit-v2.json `
  --expected-postseal-audit-sha256 1618ba74541d21f21d60bc6a3464d141d113630065074cf6169b352aa9857663 `
  --output-root eval_results\matched_eval_100\locked-semantic-global-terminal-terra-answer-v2-r7-network-recovery1\validator-v4-r1 `
  --expected-validator-run-sha256 9378772bce6d575660a27c243405d564cb03a5922e1b484f7818a4d730616d99
```
