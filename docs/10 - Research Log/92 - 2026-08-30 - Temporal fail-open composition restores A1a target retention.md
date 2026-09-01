# Temporal fail-open composition restores A1a target retention

Date: 2026-08-30

## Result

The first A1a relevance sieve was not promoted. It retained 76/381 selected
leaves but lost the smoker fact needed by `gpt4_8279ba03`, producing a sealed
post-seal result of 25/26 semantic atoms and 28/29 target-bearing leaves. The
loss was upstream of packing: the generic Terra classifier saw the dated
question and each summary, but not the authenticated date attached to each
selected leaf.

A provider-free temporal fail-open voter now composes after that sealed LLM
classification. It derives the temporal target only from the dated question
and question-only typed operator. It derives evidence dates only from the
already selected leaf's authenticated `date:YYYY-MM-DD` boundary label. A leaf
inside an exact target day or lookback interval may veto a generic
`definitely_irrelevant` decision, changing it only to `unresolved`. The voter
cannot mark a leaf irrelevant, cannot add a leaf outside the fixed union, and
does not read references, answers, semantic-atom manifests, source allowlists,
question IDs, or ordinals.

The successor passes the same isolated post-seal gate:

- **26/26 semantic atoms retained**;
- **29/29 target-bearing question/handle leaves retained**;
- **zero target-bearing leaves pruned**;
- all 11 treatment and 11 renderer-matched control prompts under the hard
  8,000-token envelope; and
- zero provider calls and zero retained transformer token state.

This is a retrieval/selection **GO**, not an answer-accuracy result. No A1a
answer or Sol judge call was made at this gate. The separately sealed A1b
compiler lifecycle described below was authorized only after the clean overlay
passed independent provenance review.

## Why this is Graphiti-like

The fix gives temporal provenance the role it has in a temporal knowledge
graph: the fact text and its valid time are separate evidence fields, and a
query-derived time relation can preserve a locally ambiguous fact for later
reasoning. The implementation does not require a graph-database rewrite. The
existing selected H leaf is the episode/fact carrier, its authenticated date
is the temporal edge, and the generic classifier remains a separate semantic
vote.

The composition rule is asymmetric by design:

```text
fixed retrieval union
        |
        +-- generic semantic classifier -> R / I / U
        |
        +-- temporal specialist ---------> positive fail-open veto only
                                             I -> U when target-time matched
        |
        `-- retained population = R union U
```

This preserves the project rule that topical or boundary metadata cannot be
the sole authority for exclusion. Temporal provenance can prevent an unsafe
exclusion; it cannot cause one.

## Sealed artifacts

| Artifact | SHA-256 |
| --- | --- |
| Base Terra dispositions/replay | `652b5f441f402d590e07bfb21130a436c8acb5666f0ac9b48bd657bce12ced5f` |
| Clean temporal effective dispositions/replay | `40a584d6499f3682a89cab1aa272c34a8ccf7ead825d2451192bc2b49114a278` |
| Clean temporal report/replay | `405a39e95ff449218e8416f8205ad2a4bb5c546bea2670efa58888932eb64a69` |
| Rebuilt actionable A1 construction/replay | `d9071196d57fedf96516aae38dfe5ed0adb5218858bee32d7f7904353c9c4da1` |
| A1a treatment/control runtime/replay | `e5d276937a98b54747d98d9790eccf4be1fea33421a43111b626445eb63ad2ce` |
| Post-seal target-retention audit/replay | `02d1a6f8af324c2a68ffdcd04d1d67172e256b4bdaadc47489e5076f62f8abd7` |
| A1b compiler preflight | `5b70afa9bb606d906fbc792d8c5779cec92eb19477cb444e8de5947f4cf1e234` |
| A1b provider release | `555ed9df14dcd66872e4e7f047beb816e6a81138c5a4a4790cd276544546e6a6` |
| A1b compiler outputs/replay | `9782c2660eb9f5aed918bdb6e0b95eeaedef68913ca2292a26835905cb1e52e0` |
| Materialized A1 construction/replay | `0da8ae97dd4931f90e4617b9dc09fb7cf99bbf3278e8e9e210f373c73ff52585` |

The earlier in-place envelope (`eb84f990...`) and its downstream
`0c1c6bc...`, `8da0cad7...`, and `8da13daa...` artifacts were withdrawn before
any compiler provider release. Independent review found that the envelope
copied the base provider-response array while rewriting effective top-level
dispositions, so a consumer could not authenticate one internally consistent
classifier history. The clean successor keeps the base response immutable and
records every I-to-U change as a separate transition re-derived from both
sealed parent pairs.

The disposition and report pairs are under:

```text
eval_results/matched_eval_100/locked-r7-after-union-a1-preflight-v2/
  terra-classifier-v1-network-recovery1/temporal-fail-open-effective-v1/
```

The new A1a runtime is under:

```text
eval_results/matched_eval_100/
  locked-r7-a1a-raw-retained-terminal-temporal-effective-v1/
```

The corresponding isolated audit is under:

```text
eval_results/matched_eval_100/
  locked-r7-a1a-postseal-target-retention-temporal-effective-v1/
```

## Scope and density

Three of the 11 questions had an executable exact-day or lookback target. The
specialist protected 56 selected leaves and changed 47 sealed I decisions to
U: 19 for the two-month jewelry window, 19 for the one-month plant window,
and 9 for the exact smoker day. Those counts are question-derived and apply to
every leaf on the authenticated target date/window; there is no
handle-specific exception.

The retained population changes from 76/381 to **123/381**:

| Metric | Base A1a | Temporal fail-open A1a |
| --- | ---: | ---: |
| Retained leaves | 76 | 123 |
| Pruned leaves | 305 | 258 |
| Leaf retention ratio | 19.95% | 32.28% |
| Retained payload token proxy | 9,054 | 14,704 |
| Fixed-union payload token proxy | 39,027 | 38,933 |
| Maximum treatment prompt | 1,588 | 3,181 |
| Maximum renderer-matched control | 4,261 | 4,223 |
| Target semantic atoms | 25/26 | **26/26** |
| Target-bearing leaves | 28/29 | **29/29** |

The slight fixed-union proxy difference comes from its authenticated
per-evidence disposition field: the successor spells 47 rows `unresolved`
instead of the longer `definitely_irrelevant`. It does not reflect a different
381-leaf selected population; the selected-population SHA remains
`a201012a536d4d9816c2756ee0f91a37246646002b563e94595b9d2b06af401c`.

The repaired A1 fact-compiler preflight derives **21** actionable Terra calls,
up from 16 in the over-pruned arm. Its compiler prompts range from 692 to
2,241 tokens before their output reserve, so the additional temporal evidence
does not approach the hard cap. After the clean successor passed review, all
21 calls completed through the local Terra endpoint. Strict materialization
authenticated 21/21 response checkpoints and sealed byte-identical compiler
outputs. The compiled A1 correctly remains `materialized_with_unresolved_closure`:
facts cover some retained leaves, while ambiguous retained leaves remain
explicitly unresolved and must be delivered raw to the terminal reader.

## Implementation

- `tools/matched_eval/r7_after_union_temporal_fail_open.py` authenticates the
  exact A1 and base-classifier construction/replay pairs, replays semantic
  selection, derives temporal targets, performs the one-way I-to-U transition,
  and emits a separate gold-blind effective overlay and report.
- `tools/run_r7_after_union_temporal_fail_open.py` seals both construction and
  byte-identical replay artifacts without provider IO.
- `tests/test_matched_eval_r7_after_union_temporal_fail_open.py` covers exact
  target-day rescue, non-temporal identity behavior, deterministic replay,
  impossible dates, foreign requests/populations/leaf receipts, incomplete
  coverage, and firewall tampering.

The final independent temporal composition and downstream-consumer suite
passed 110 tests:

```text
.pixi\envs\dev\python.exe -m pytest \
  tests\test_matched_eval_r7_after_union_temporal_fail_open.py \
  tests\test_run_r7_effective_overlay_inputs.py \
  tests\test_matched_eval_r7_a1a_raw_retained_answer.py \
  tests\test_a1a_postseal_target_retention.py \
  tests\test_matched_eval_r7_after_union_a1.py \
  tests\test_run_r7_after_union_a1_classifier.py \
  tests\test_matched_eval_temporal_insufficiency_specialist.py \
  --basetemp .test-tmp\root-temporal-effective-independent-20260830-a \
  -p no:cacheprovider -q
```

## Promotion boundary

The repaired 123-leaf population, not the rejected 76-leaf population, was the
input to A1b. A1b has now sealed exact-citation facts while preserving every
unresolved retained leaf explicitly. The next bounded experiment is a terminal
reader over `compiled facts + unresolved raw leaves`, deduplicated only after
the fixed union and kept under the same 8K envelope. The 26/26 result is still
only a retention gate; neither it nor successful compilation establishes QA
accuracy, and neither alters the protected 89/100 development parent.
