# Closure v9 distinguishes invoked scoring from scalar bypass

**Status:** retrieval campaign complete: 79/79 eligible question artifacts,
ten shard indexes, and the merged generation are sealed. The downstream
matched result is recorded in [Research Log 59](59%20-%202026-08-27%20-%20Independent%20closure%20v9%20matched%20outcome.md): both closure arms scored
52/100 against S0-v2's 53/100 and were rejected from positive-only
composition.

V8 published question artifacts for ordinals 3 and 4, then failed closed on
ordinal 5. That question does not request complete-set coverage, so the frozen
selector intentionally bypassed its already-bound score provider. The fresh
coverage report truthfully carried the provider's immutable identity but no
nested scoring-call timer. V8 treated every missing nested `elapsed_s` as a
malformed invocation and stopped before publishing the ordinal-5 artifact.

V9 changes only the timing-normalization contract. It does not change the
frozen retrieval implementation, eligibility predicate, protected arm
budgets, label-free structural source identity, exact route-local identities,
gold firewall, or stable-S0 receipt-linkage checks.

## Exact accepted report shapes

Every accepted fresh report must contain a numeric, non-boolean top-level
`elapsed_s`. V9 then accepts exactly one of two score-provider shapes.

### Invoked score provider

- `score_provider_report.elapsed_s` is present and is a numeric, non-boolean
  value;
- `selection_status` is not `bypassed`;
- normalization removes top-level `elapsed_s` and nested
  `score_provider_report.elapsed_s`; and
- `fresh_report_normalization_removed_fields` is exactly
  `["elapsed_s", "score_provider_report.elapsed_s"]` in that order.

A report claiming `selection_status == "bypassed"` while carrying a nested
scoring timer is contradictory and fails closed.

### Authoritative scalar bypass

The nested timer may be absent only when all of the following hold exactly:

- `selection_status == "bypassed"`;
- `bypass_reason == "not a set query"`;
- `requires_completeness is False`;
- `score_provider_fallback == ""` and `fallback_reason == ""`; and
- `score_provider_report` contains exactly `model_id`, `model_revision`,
  `checkpoint_sha256`, `device`, `dtype`, `runtime`, and
  `retained_transformer_state_bytes`.

The identity fields retain their exact scalar types, the checkpoint is a
lowercase SHA-256 digest, required runtime identity text is non-empty, and
`retained_transformer_state_bytes` is the exact integer zero. Normalization
removes only top-level `elapsed_s`, so the sealed removed-path list is exactly
`["elapsed_s"]`.

Missing or extra identity fields, a nonempty fallback, a different bypass
reason, false type substitutions such as `0` for `False`, boolean timing, or a
missing nested timer on any invoked/fallback path remains a hard failure.
Other similarly named timing fields are not stripped.

## Attestation and replay boundary

The complete fresh report is persisted and its identity must equal the fresh
predecessor receipt's coverage-report hash before normalization. The generator
records the normalized fresh-report identity and the paths actually removed.
Resume and merge validation recompute both. The matched closure adapter
independently repeats the same checks before projecting either arm.

The historical coverage-report payload was never persisted, so v9 does not
invent or claim a normalized comparison against it. The historical report
hash and its timing-derived receipt hashes remain recorded but not compared;
all stable predecessor fields, stable root-stage fields, evidence order,
provider messages, prompt, and predecessor-to-root linkage remain exact.

V8 artifacts are not silently reused. V9 has distinct eligibility, preflight,
policy, question, shard, merged-generation, and downstream artifact-role
versions under a separate `independent-closure-v9` output root.

## Verification and sealed preflight

The integrated generator and matched-adapter suite passes 141/141. Coverage
includes both accepted shapes, actual removed-path lists, input nonmutation,
contradictory bypass/timing combinations, missing and extra identity fields,
scalar-type tampering, retained-state tampering, fresh-report resealing, and
downstream adapter rejection. Independent audit gave the v9 generator and
adapter contract GO.

| Field | Sealed value |
| --- | --- |
| question population | 100 |
| eligible retrieval population | 79 |
| retrieval/provider calls during preflight | 0 / 0 |
| eligibility manifest SHA-256 | `748bd56a7efb8fd70d36bc96f099a53fc506469565577de9635908f6773bdee1` |
| preflight SHA-256 | `268cb5bfa70661de470b5142163d9447a199c05ce713233cb75ff7ce25ec4451` |

## Sealed offset-0 canary

Offset 0 completed all 7/7 eligible questions and sealed its shard artifact:

```text
77133620564510efbc2554e29fb3a587c9de9b6b02998fe803fbb1d80bd66b36
```

The matched adapter accepted 14/14 downstream arm projections: one
representative-bridge and one artifact-global projection for each question.
The fresh S0 reports exercised both v9 normalization branches, with three
invoked score-provider reports and four authoritative scalar bypasses.
Ordinal 5, the former blocker, now published and projected through the exact
identity-only bypass path.

Aggregate offset-0 structural flow was:

| Arm | Candidates | Selected before dedup | Exact-S0 exclusions | Admitted |
| --- | ---: | ---: | ---: | ---: |
| representative bridge | 291 | 39 | 0 | 39 |
| artifact global | 1,696 | 66 | 25 | 41 |

## Sealed shard 10

Shard 10 then completed its 7/7 eligible questions and sealed at:

```text
324ef35e4d835d6b0e1dedae82fc1d6ae63be1e9f99e9f3968fa74dbe9517873
```

Independent downstream validation accepted all 14/14 arm projections. Its
structural flow was:

| Arm | Candidates | Selected before dedup | Exact-S0 exclusions | Admitted |
| --- | ---: | ---: | ---: | ---: |
| representative bridge | 403 | 52 | 0 | 52 |
| artifact global | 1,736 | 66 | 21 | 45 |

## Completed shard ledger

All remaining shards completed after the two canaries. The sealed retrieval
frontier is:

| Shard offset | Eligible questions | Shard SHA-256 |
| ---: | ---: | --- |
| 0 | 7 | `77133620564510efbc2554e29fb3a587c9de9b6b02998fe803fbb1d80bd66b36` |
| 10 | 7 | `324ef35e4d835d6b0e1dedae82fc1d6ae63be1e9f99e9f3968fa74dbe9517873` |
| 20 | 8 | `39ee51a5372735a0d98a0637b8e095157d94c664c65729661f5453fbeb31097d` |
| 30 | 7 | `62c56a62a141de5103b0f72275103ced54242c4874364038861c9188d1110eeb` |
| 40 | 8 | `3eb1f391debc2fc8871037a40d07e22a6db0d792c80beec5a8fee97bd4ae4c31` |
| 50 | 8 | `723ccc8af286a6e8c5872b8a7d67ce103a183bf316e9aa8cc8d5a9fb2b540883` |
| 60 | 8 | `d0f9b3611031c6f40de2f139476cd88ba870582ed8440dbc52c7e489af14f166` |
| 70 | 9 | `6a7f4d4d5fcf0f7c91daa654e361b3d05f062184de5e5002d1517af219a2a9d7` |
| 80 | 8 | `743b45386aba01db8e7c41f5c0b9f309238f407e4de13f3e35ebb52b94d39c75` |
| 90 | 9 | `d9395c3327df24bf7fce05ac73cfdc1c2476369c7f1d91095782288bfbdb8911` |
| **sealed total** | **79/79** | -- |

These are retrieval membership and provenance counts, not relevance or
answer-accuracy scores. The merged generation sealed at
`cf541c40f0749dcf9e436080c56dcf251232fd9ac7c844be49e2dfd8764a7ee5`.
All question and shard sidecars, matched arm projections, answer/judge
lineage, and zero-call replays passed. Research Log 59 contains the exact
downstream seals, budgets, target funnel, calls, and negative paired result.
