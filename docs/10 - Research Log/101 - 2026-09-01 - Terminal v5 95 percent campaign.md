# 2026-09-01: terminal-v5 95% campaign

## Objective

Run the next comparable locked LongMemEval-S validation campaign after the
orphaning repair. The policy under test is terminal compilation mode
`v5-linked-backfill`. The validation claim requires at least 95 accepted Sol
judgments over the sealed 100-question population spanning ten resident
approximately 1M-token namespaces. The exact-11 assay is a promotion canary,
not a 95% score.

The operational procedure is
`docs/02 - Implementation/13 - 95 Percent Full100 Campaign Playbook.md`.
Historical v2/v3 roots remain frozen.

## Comparison boundary

- strongest comparable development parent: V3, 89/100;
- historical terminal V2: 79/100;
- prior linked exact-11 diagnostic: 6/11, versus the earlier 5/11 best;
- fresh terminal-v5 score at campaign start: none;
- required improvement over the comparable parent: at least +6 net accepts.

The unrelated historical closure-selector label `V5` is not this terminal-v5
compilation policy.

## Gate 0 snapshot

- Git HEAD: `2124f98c3f6c7fcea0bcac919d5aa72a4bc9fbc4`.
- Snapshot date: 2026-09-01, America/Los_Angeles.
- `LITELLM_KEY`: populated; value not printed.
- campaign Python modules: `py_compile` passed.
- `git diff --check`: passed.
- exact-11 v5 root: absent.
- resumable full100 v5 root: absent.
- Terra-answer v5 root (and nested judge root): absent.
- provider calls before promotion: zero.

Dirty inventory captured before this log file was created:

```text
 M docs/03 - Architecture/00 - System Overview.md
 M docs/05 - Standards/00 - MC-STD-DATA-v0.md
 M src/memory_condense/application/condenser.py
 M src/memory_condense/application/ingest_workflow.py
 M src/memory_condense/application/retrieval_workflow.py
 M src/memory_condense/associations/association_repository.py
 M src/memory_condense/ingest/validator.py
 M src/memory_condense/persistence/db.py
 M src/memory_condense/persistence/memory_store.py
 M src/memory_condense/persistence/transcript_store.py
 M src/memory_condense/search/indexes/index_lifecycle.py
 M src/memory_condense/search/indexes/lexical.py
 M src/memory_condense/search/indexes/span_source_queries.py
 M tests/test_compiled_cache.py
 M tests/test_condenser.py
 M tests/test_db.py
 M tests/test_lexical.py
 M tests/test_matched_eval_semantic_global_terminal_adapter.py
 M tests/test_matched_eval_typed_additive_composer.py
 M tests/test_memory_store.py
 M tests/test_retrieval.py
 M tests/test_run_locked_semantic_global_terminal_full100_answer.py
 M tests/test_run_locked_semantic_global_terminal_full100_construction.py
 M tests/test_run_reduced_semantic_binary_search_assay.py
 M tests/test_run_reduced_semantic_global_terminal_assay.py
 M tests/test_run_reduced_specialist_retrieval_assay_v3.py
 M tests/test_transcript_store.py
 M tests/test_validator.py
 M tools/matched_eval/semantic_global_terminal_adapter.py
 M tools/matched_eval/typed_additive_composer.py
 M tools/run_locked_semantic_global_terminal_full100_construction.py
 M tools/run_locked_semantic_global_terminal_full100_resumable.py
 M tools/run_reduced_semantic_binary_search_assay.py
 M tools/run_reduced_semantic_global_terminal_assay.py
 M tools/run_reduced_specialist_retrieval_assay.py
 M tools/run_reduced_specialist_retrieval_assay_v3.py
?? docs/02 - Implementation/13 - 95 Percent Full100 Campaign Playbook.md
?? docs/08 - Analysis/29 - Orphaning audit and lifecycle repair 2026-09-01.md
?? docs/10 - Research Log/100 - 2026-09-01 - Orphaning audit and lifecycle repair.md
?? src/memory_condense/persistence/pending_ingest_schema.py
?? src/memory_condense/persistence/pending_ingest_store.py
?? tests/test_pending_ingest_store.py
?? tools/run_reduced_specialist_retrieval_assay_v4.py
```

## Provider-free defect found before construction

The first terminal-v5 exact-11 attempt failed before publication and before
any provider call. `_post_dedup_backfill` correctly expanded the final row
population, but final-fit authority still compared those rows only with the
pre-backfill `DeduplicationReceipt`. Any real v4/v5 admission therefore failed
closed with `terminal final fit received rows outside its authenticated dedup
population`.

The repair threads the exact `PostDedupBackfillReceipt` through typed prompt
compilation and authenticates:

1. its initial-dedup hash against the dedup receipt;
2. the ordered final population as dedup survivors plus admitted rows;
3. the actual final row order against that sealed population; and
4. plane-selection authority when available.

Legacy no-backfill behavior and dedup authority transfers remain intact. The
focused public compiler regression exercises a genuine admission in both v4
and v5, preserves inherited G hard authority on the retained R row, and fails
closed on population, dedup-link, and receipt tampering.

Regression evidence before the canary retry:

- focused final: 1 passed in 38.50 s;
- v4/v5 compatibility selection: 2 passed, 12 deselected in 37.99 s;
- adapter plus v61 contract: 15 passed in 232.33 s;
- `git diff --check`: clean.

## Second gate failure: frozen R7 predates evidence conservation

The repaired exact-11 run completed three resident namespaces, then failed
closed on ordinal 94 before publishing any v5 artifact:

```text
R7 question 94 differs from exact store/search replay
```

This was not store, index, vector, query, backfill, linker, or dirty-worktree
drift. The store/index lifecycle and compact query commitment remained exact.
The first material difference was upstream R7 search:

| Metric | current HEAD | frozen R7 |
|---|---:|---:|
| classifier calls | 1,505 | 1,501 |
| retained leaves | 753 | 709 |
| pruned leaves | 0 | 44 |
| attempted segments | 7,634 | 7,559 |
| packed evidence rows | 12 | 12 |
| complete chat plus output tokens | 3,731 | 3,727 |

Commit `2124f98` intentionally made the R7 residual classifier fail open:
`required_role_absent`, `exact_literal_absent`, and `dual_gate` remain audit
reasons but no longer claim authenticated impossibility. The 44 formerly
pruned leaves therefore survive and contribute 75 additional attempted
segments. The frozen R7 artifact was built with the older heuristic-pruning
semantics.

The behavior change had not received a new mechanism or policy identity, so
the old parent and current search shared nominal identifiers despite producing
different bytes. The campaign must not bypass the equality gate or restore
legacy pruning merely to satisfy the old seal. The approved treatment is:

1. retain an authenticated legacy mode solely for historical replay;
2. give evidence-conserving fail-open search a distinct policy/mechanism
   identity and sealed semantics field;
3. rebuild R7 under a new provider-free root using the unchanged sealed gate,
   vectors, stores, and query population;
4. rebuild the dependent V7/exact-terminal/full100 lineage; and
5. keep all provider releases blocked until the new promotion lineage seals.

Provider calls consumed by both failed canary attempts: zero. Neither attempt
published a terminal-v5 artifact.

## Campaign results

The evidence-conserving R7 successor sealed at 2026-09-02 00:35:54 PDT with
SHA-256
`6cd26b55092d0a93aca1afc5209874a1bb7ebf7927a805e4f2d8b274fb48f8e3`.
Full100 preflight sealed at 01:48:24 with SHA-256
`c8373ef198fc5b360f9da70c0c6b366fd93aef01280adfc4dd6243ca51ae8277`;
construction sealed at 04:53:24 with SHA-256
`57e162240bbaae0470e0b102e2b32a547b550088c87340dbe155de2218cf8c00`.
It contains 100 ordered questions, 68 terminal plans, 32 V3 passthroughs, and
ten namespace checkpoints. Provider calls consumed: zero.

Construction wall time was 3 h 05 m. The ten final namespace artifacts and
sidecars occupy 2.342 GiB; the ten resumable checkpoints and sidecars duplicate
another 2.342 GiB, for a 4.684 GiB validation footprint. The cost is serial
provider-free namespace replay and duplicated audit-payload I/O, not Terra or
Sol latency. The next performance refactor should store the full audit payload
once and have checkpoint/final manifests reference it rather than duplicating
the bytes.

There is no separate exact-11 construction in the streamlined route. The
promotion audit projects the eleven fixed plans directly from authenticated
full100 sidecars. Full100 replay completed at 2026-09-02 11:54:48 PDT after
19 m 18 s and reproduced the construction byte-identically at
`57e162240bbaae0470e0b102e2b32a547b550088c87340dbe155de2218cf8c00`.
It authenticated all ten checkpoints, made zero provider calls, and retained
zero transformer-token state.

The direct promotion audit sealed at 2026-09-02 12:07:27 PDT after 12 m 17 s
with SHA-256
`65285b9db760cb649e621465492ff0c323c9449c0b2735a34dd8bd70f23cf369`.
It proved all 26/26 semantic atoms provider-visible and usable across the fixed
eleven-question projection; source target coverage was also 26/26. It made zero
provider calls and retained zero transformer-token state. The provider-free
promotion gates are complete.

A zero-completion `GET /v1/models` check then reached the controlled gateway,
returned 13 model entries, and verified `max_retries=0`; no model completion
was requested. Terra answer preflight sealed at 2026-09-02 12:19:37 PDT after
10 m 47 s with SHA-256
`0c4464cf288b93f814991fd7abc2d74d76c5ce7396ae8829eeac43d1ec38f289`.
It binds 100 questions, 68 synthesis prompts, 32 passthroughs, a maximum
complete prompt envelope of 7,995 tokens, and exactly 68 authorized provider
calls. Physical provider calls at preflight: zero.

The Terra provider release sealed at 2026-09-02 12:29:34 PDT after 9 m 36 s
with SHA-256
`2b9bb5741afe18e4b9c631b0e6ec2bb4d4dd2ee5e1d6f3b5630b70cbb5b4a5d7`.
It remained gold-blind and made zero physical calls. The execution environment
then required a fresh explicit approval for this exact 68-call payload despite
the earlier general local-LiteLLM permission; the rejected launch made zero
provider calls and did not create completion checkpoints.

After explicit approval, all 68/68 Terra calls completed successfully. The
first request was written at 2026-09-02 12:58:38 PDT and the final response at
13:02:38, approximately four minutes of provider wall time at concurrency four
and zero retries. Every request carried a locked question and its sealed
evidence/facts without gold. Materialization used all 68 checkpoints, made zero
additional provider calls, produced 100 ordered predictions, and changed seven
predictions from the V3 parent. The answer-run SHA-256 is
`f1d774e98f48758b8ced70be05064e0af0aa538f9673f7744f0df8607ba54946`.

Answer replay was byte-identical and sealed at SHA-256
`2cbb053f31c2ba713a9fa16819b4bd8d007ff5b758da99e919b4c6d8795f1d41`.
Sol judge preflight and release then sealed at
`34099ddb56fa2c2e2ba5d42c50cbe4cb142c16e94a95ed4183d91b507f80c8b9`
and `aea418d667daec86899636aaebfec09406f67123df6db994547769aca0e83573`.
All 100/100 authorized Sol calls completed from 13:14:15 through 13:19:44 PDT
at concurrency four and zero retries. Each prompt contained only the locked
question, reference answer, and sealed Terra prediction.

The judge and replay were byte-identical at
`edccbd49a20bf92fcb52306fe28557eeccb8ebba69e9e12a26d5d6cc5d530239`.
The deterministic score and score replay were byte-identical at
`91ae36ebb7ef48fb914f7236ca03998adb0b22f58d98c29bfa8ecccd3739dce1`.
The terminal-v5 campaign scored **88/100**. It missed the 95% target and was
one net accepted answer below the comparable V3 parent score of 89/100.

## Result delta and failure shape

The answer policy changed exactly seven predictions. Comparing the sealed V3
and terminal-v5 Sol judgments gives one rescue, two regressions, and four
correctness-neutral rewrites:

| Ordinal | Question ID | V3 | terminal-v5 | Effect |
|---:|---|---:|---:|---|
| 5 | `06878be2` | correct | incorrect | regression: Sony evidence was replaced by incompatible Nikon/Canon details |
| 27 | `6b7dfb22` | correct | correct | neutral rewrite |
| 36 | `32260d93` | correct | incorrect | regression: stand-up preference was replaced by a movie musical |
| 49 | `a89d7624` | incorrect | incorrect | neutral miss |
| 54 | `gpt4_8279ba03` | incorrect | correct | rescue: `I don't know` became `You bought a smoker` |
| 81 | `1a1907b4` | correct | correct | neutral rewrite |
| 82 | `1d4e3b97` | incorrect | incorrect | neutral miss |

The arithmetic is exact: `89 + 1 rescue - 2 regressions = 88`. All 93
unchanged predictions retained the same correctness judgment, so the decline
is not Sol judge drift.

Of the 68 eligible Terra rows, the validator accepted seven replacements,
kept the parent on 57 valid completions, and failed safely to the parent on four
invalid completions. The other 32 rows were sealed V3 passthroughs. The four
invalid reasons were two `aggregate_scope_incomplete`, one
`required_slot_coverage`, and one `typed_list_entailment`.

The twelve remaining misses are concentrated by demand:

| Demand | Count | Ordinals |
|---|---:|---|
| multi-session numeric reduction | 7 | 14, 28, 40, 53, 67, 69, 97 |
| single-session preference synthesis | 4 | 5, 36, 49, 82 |
| temporal timeline reasoning | 1 | 94 |

All seven numeric-reduction misses retained their parent predictions. The next
iteration therefore needs a bounded numeric set/count reducer and stricter
replacement arbitration for preference synthesis; simply widening retrieval
again would not address the observed failure shape.

## Proof-carrying policy follow-up (2026-09-02)

The follow-up replaced answer-policy guessing with finite proof search. For a
finite sealed memory, a closed relevant frontier, and a versioned deterministic
typed operator, the policy can compute the answer exactly and replay its proof.
That guarantee is conditional on the formalized grammar and frontier-closure
certificate; it is not a completeness guarantee for arbitrary natural-language
questions or incomplete memory. When closure cannot be certified, the policy
keeps the protected parent rather than manufacturing an answer.

The provider-free v2 numeric frontier is rooted at
`eval_results/matched_eval_100/locked-full100-numeric-frontier-v2` and its
materialization and replay are byte-identical at
`15a7d9bbd90666f441ed93089ef331d86497e569b59200eb52248a82bc231566`.
Q28 closed under the certified census. Q14, Q53, and Q69 remained open: Q14
had a cuisine-identity disagreement between the full-store census and the
packet, Q53 exposed additional plant identity/state variants, and Q69 could
not certify the boundary between current and unknown obligation state. These
are correct refusals, not missing arithmetic.

The receipt-bound `policy-v5-r2` overlay sealed as run
`cb19ee0649ab50f55ca6db42d9333bf881f3434cfa754449a9ed4da3fd1b9e84`
and replay
`ec63ff495f86c48548e1490fb24dd87b8136a22990263a584ae8896fdd4186bb`.
Its differential Sol plan sealed at
`025a14c8e3191019c5fd66399f847f8b4e901c88ac9722abda72dd50bcad51b4`.
The novel-call preflight and release sealed at
`83267b1e4623a84ae946927929989c55d7186626ca0991f9e6082e54058e7358`
and
`1c4675011a0e1c1e3b703110a933dfa65b5ea576a73be9ce4a192c25f8c710f3`.
Exactly two Sol provider calls were made, for Q28 and Q97; both were judged
correct. The novel judge and replay are byte-identical at
`75c687a20a4a9fca4ec7f33add823d1bd428daebe595bc4818c79f108179dd9c`.

Merging those two judgments with the 98 exactly reusable historical judgments
produced a final **92/100**. The sealed merge SHA-256 is
`e20286f2b8d9e81e4b69dd947b59d7e111c2b47842f3a54b15e95c668e001f3c`.
The remaining gap is no longer primarily a numeric-execution problem. The next
missing primitive is certified topic, entity-identity, state, and component
boundaries so that a local evidence neighborhood can be proven to represent
the complete global answer set.

## Operator-material-v3 validation pass (2026-09-02)

We implemented the next provider-free frontier profile. It preserves raw state
through candidate admission, then normalizes only admitted operands to the
reducer-observable state `operator_eligible`. Jewelry and museum/gallery were
added to proof applicability. The loader was streamlined to authenticate the
common sealed query/retrieval population once while still hash-checking and
constructing each namespace store independently. Its verifier now also rejects
missing or duplicate lifecycle partitions, incorrect per-namespace row lists,
window-index mismatches, and non-normalized v3 census atoms.

The assay produced seven frontier rows and closed four. Q28 remained closed;
Q53, Q67, and Q69 newly closed. Q14, Q40, and Q77 stayed open, so the change did
not force unresolved material mismatches through the operator. The frontier
materialization and replay were byte-identical at
`94092dcd879a3869f63177a08bd9366f7221bbed3d2fa33da7b268bb16ca6f59`.
Gold loaded: false. Provider calls: zero.

The `policy-v5-r3` overlay sealed as run
`a145c8d6d5587293347621c5ca32d367e9aefe050c706e7232691a6c49aa34a9`
and byte-identical replay
`ec0672539d5a4d8df33673896a7c07bb8b0052a871cae7df7c66851e35f55052`.
Its provider-free differential plan sealed as
`6df257b380cd6f4d19dac785cb85017766b1f8fdfe5561abd10b445b4a45f39d`,
reused 97 authenticated judgments, and exposed exactly Q53, Q67, and Q69.

The novel-call preflight and release sealed at
`640d2b324e425ac3d679aff5400162207c9b51adb213276548d8b9555f20f053`
and
`9eed49a96a6167f180224adb6abe5bb41457c475e2b3a58dd19a7a9dc9aae264`.
Exactly three authorized Sol calls ran with zero retries. All three were
accepted. The novel judge and replay are byte-identical at
`dc5d145cb422203b08ba4ee14b2ee9dad54c6f3d71bde6dcedc5a9608a9355ef`.

Merging the three novel judgments with the 97 exact historical reuses produced
the first sealed validation pass: **95/100**. The merge SHA-256 is
`aa210a8bba87897d7fc8e3f4e2a7e71cbcc929fa4eeac6ce5cbf6ef56567c952`.
The five remaining misses are Q14, Q40, Q49, Q82, and Q94. The validation100
threshold is met; the policy must now be frozen before an untouched
confirmation200 campaign.
