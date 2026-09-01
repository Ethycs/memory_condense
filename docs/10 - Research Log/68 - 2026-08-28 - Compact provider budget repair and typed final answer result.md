# Compact provider budget repair and typed final answer result

**Date:** 2026-08-28

**Status:** implemented, sealed, replayed, and independently judged on the
locked 100-question population; **73/100**, below the 95% target

## Result

The compact-budget typed-final arm scored **73/100** under a fresh full-100 Sol
semantic judgment. Its immediate locked parent scored 72/100. The net measured
gain is therefore one question, not a 95% result and not evidence that packing
alone solves the remaining recall problem.

The result is nevertheless structurally useful. It closes a real serialization
bug, demonstrates an exact 100-question answer-and-judge path over ten
independently ingested approximately one-million-token memories, and separates
terminal packing failures from retrieval, lane selection, relation formation,
and answer-validation failures.

## Serialization-boundary defect

The first typed-final composition budgeted the verbose canonical
`TypedEvidencePacket.provider_projection()` during fair premerge, while the
actual Terra prompt used `compact_typed_evidence_projection()`. The budget was
therefore charged against bytes that were never submitted to the provider.
This prematurely removed 67 evidence items from five questions even though the
terminal prompt fitter removed no items and the largest complete prompt
envelope was only 4,652 of 8,000 tokens.

The repair introduced an explicit provider payload mode:

- `CANONICAL` preserves the existing projection and receipt identities;
- `COMPACT_FINAL` makes fair-premerge accounting use the exact compact schema
  that the final answer prompt will serialize; and
- packet subsets preserve the selected mode through fair merge and hard fit.

Canonical local receipts intentionally omit the new mode field, preserving old
artifact identities. Compact mode has a distinct provider projection and token
accounting identity. The compact provider payload still contains typed facts or
exact retrieved chunks, opaque evidence and story handles, method origin,
frontier state, and validation metadata; local source locators remain sealed
outside the provider prompt.

Focused regression coverage passed **62 tests**. It includes a case where the
canonical representation exceeds the budget while the compact final payload
fits, and verifies exact summary text, provenance bindings, token accounting,
receipt separation, and absence of raw locators. The expanded target-flow
analyzer passed its focused **9-test** suite.

## Sealed compact lineage

| Artifact | SHA-256 |
| --- | --- |
| full-store closure input | `044e60f308287dda4d87106646e4cc56f0e96d513b2bfd03a7473da9994ef5c4` |
| compact typed composition | `21be1ebfe628eae55dd543312e59c315f08de298b9d1895fc757b6517f869933` |
| Terra preflight | `b5f951e56393513543f046c9cd454d323528f8a3d4c4f3150ad037503d7bf1a2` |
| Terra answer run | `ce81033e0658fcf2706e95214cfe29323f4c84adb5ce3deb96f8da79ceb34907` |
| Terra replay report | `117ff8ea1d7f1745263ec90ae2d13ba13f2a9814defaac6bfb435c7421a82a61` |
| target-flow assay v2 | `13aee100845e2c2ed906163c752172de65ac4e8b1f22c0a5a71acc0aac211cad` |
| Sol judge preflight | `35d0d6309c341651059238b2e473727487784aabc578d12c1a8fbd83434819d8` |
| Sol full-100 judgment | `7ddbfe25e1f048e44524fb948d29463d9393c6a8b0fdee6c62cd0bc965f295e0` |
| score ledger | `34a1cfff13acf00170c101db9e37490d3c3ef3b607698a89021519362f1f2b1a` |

The compact lineage reuses the byte-identical full-store closure input. Its
composition and preflight were constructed with zero provider calls, no gold,
and zero retained transformer token-state bytes.

## Packing effect

The repaired evidence flow is:

```text
3,591 retrieved local items
    -> 3,591 after exact post-selection dedup
    -> 1,887 retained by method lanes
    -> 1,887 retained by fair merge
    -> 1,887 retained by final hard-cap fitting
```

Compared with the first sealed composition, the compact arm restores all 67
items previously lost at fair merge:

| Ordinal | Additional final items |
| ---: | ---: |
| 5 | 2 |
| 27 | 15 |
| 36 | 21 |
| 43 | 7 |
| 49 | 22 |

The largest input prompt rose from 3,884 to 5,304 tokens. With the fixed
768-token output reserve, the largest complete envelope is 6,072 of 8,000.
No terminal hard-fit drop occurred.

The post-hoc target assay did **not** improve: 52 of 84 declared target
components reached the global prompt boundary in both compositions. The source
target funnel also remained 46 retrieved, 41 lane-selected, and 41 globally
bound. The restored 67 items therefore prove that projection-tax loss existed,
but they do not recover the five source targets already lost inside method
lanes or the 13 source targets never discovered.

## Live answer and judge protocol

The sealed Terra prompts contain dated question, protected parent fallback,
typed evidence, opaque provenance handles, story links, and validation
contracts. They contain no reference answer or target registry. Terra is the
actual downstream LLM consuming the memory substitute context; it is not a
synthetic scoring shim.

The authorized live campaign produced exactly:

- 100 Terra request journals and 100 Terra response journals using
  `codex_sdk/gpt-5.6-terra`, with zero retries and no runtime gold; and
- after sealing those predictions, 100 Sol request journals and 100 Sol
  response journals using `codex_sdk/gpt-5.6-sol`, each carrying question,
  reference, and the sealed Terra prediction.

Checkpoint-only materialization then used all 100 Terra responses with zero
additional physical calls. Terra replay was byte-identical to the answer-run
SHA. Sol materialization and replay likewise used the 100 sealed responses with
zero additional physical calls; the replayed judgment and score ledger had the
same SHA values shown above.

An earlier sandboxed Terra attempt was denied by the operating system before a
TCP connection was made. It left four request files and no responses. Those
files were moved, without deletion, to
`sandbox-blocked-request-quarantine-20260828T002616` and are not counted as
provider completions.

## Answer decisions and score movement

Terra changed 19 of the 100 protected parent predictions:

| Outcome against the immediate 72/100 parent judgment | Count |
| --- | ---: |
| wrong to correct | 5 |
| correct to wrong | 4 |
| correct to different correct | 6 |
| wrong to different wrong | 4 |

All 100 parent prediction hashes matched the immediate parent judge artifact.
There were no Sol verdict changes among the 81 byte-identical unchanged
predictions. The observed +1 is therefore exactly the net of five rescues and
four regressions, rather than judge drift on unchanged answers.

The completion validator produced 19 accepted replacements, 15 explicit
`keep_parent` decisions, and 66 fail-closed parent fallbacks. Of the 66:

- 61 were `replace_contract` rejections because Terra labeled an answer as a
  replacement while returning text exactly equal to the protected parent;
- one each was rejected for required-slot coverage, scalar-advisory
  disagreement, typed numeric entailment, typed numeric prediction, and typed
  text entailment.

The 61 same-parent replacement rejections preserved 52 already-correct and nine
already-wrong parent answers, so they did not change this score. Across all 66
fail-closed fallbacks, 54 correct parents were protected and 12 wrong parents
remained wrong. This distinguishes a noisy decision-label contract from actual
retrieval abstention: most invalid rows proposed no textual answer change at
all, while five rows exercised substantive evidence validation.

## Interpretation and next decisions

1. Keep the compact-budget repair. It aligns budgeting with submitted bytes and
   removes a genuine dataflow defect without weakening the hard prompt cap.
2. Do not spend another live 100+100 campaign on packing alone. Target coverage
   stayed fixed and the measured gain was only one point.
3. Add a provider-free surplus-fill phase after the non-borrowable method
   minima. It should use remaining global capacity to rescue the five exact
   source targets already retrieved but lost inside their lane rankings.
4. Add query-specific local-to-global expansion for the 13 never-retrieved
   source targets, concentrating on dispersed temporal joins, numeric/multi-row
   questions, preference traces, and representative-event selection.
5. Strengthen relation formation and CAV-style reinjection. Only 10 of 23
   required relations currently reach the prompt with all operands and an
   explicit link.
6. Canonicalize a same-text `replace` as `keep_parent` for cleaner telemetry,
   while retaining fail-closed validation for the five substantive rejection
   classes. This cleanup is not expected to improve accuracy by itself.
7. Evaluate the next sealed arm by the same two-stage protocol and compare
   answer deltas against the exact parent hashes before authorizing another
   full judge campaign.
